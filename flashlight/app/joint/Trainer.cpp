/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/joint/Trainer.h"
#include "flashlight/app/joint/Flags.h"

using namespace fl::ext;
using namespace fl::lib;

namespace fl {
namespace app {
namespace joint {

#define FL_APP_JOINT_VERSION "0.1"

/* ============= Public functions ============= */
Trainer::Trainer(const std::string& mode) {
  experimentDirectory_ =
      fl::lib::pathsConcat(FLAGS_exp_rundir, FLAGS_exp_model_name);

  if (mode == "train") {
    initTrain();
  } else if (mode == "continue") {
    initContinue();
  } else if (mode == "fork") {
    initFork();
  } else if (mode == "eval") {
    initEval();
  } else {
    throw std::invalid_argument("Trainer doesn't support mode: " + mode);
  }
  checkArgs();
  gflagsStr_ = fl::app::lm::serializeGflags();
  FL_LOG_MASTER(INFO) << "Gflags after parsing \n"
                      << fl::app::lm::serializeGflags("; ");

  initArrayFire();
  if (FLAGS_distributed_enable) {
    reducer_ = std::make_shared<fl::CoalescingReducer>(1.0, true, true);
  }
  FL_LOG_MASTER(INFO) << "[Log directory] " << experimentDirectory_;
  if (isMaster()) {
    dirCreate(experimentDirectory_);
    logWriter_ = createOutputStream(
        pathsConcat(experimentDirectory_, "log"), std::ios_base::app);
  }
}

void Trainer::runTraining() {
  FL_LOG_MASTER(INFO) << "Training started (asr-epoch=" << asrEpoch_
                      << ", lm-epoch=" << lmEpoch_ << ", batch=" << batchIdx_
                      << ")";

  fl::allReduceParameters(encoder_);
  // fl::allReduceParameters(asrFrontEnd_);
  // fl::allReduceParameters(lmFrontEnd_);
  // fl::allReduceParameters(asrCriterionLinear_);
  fl::allReduceParameters(lmCriterion_);
  auto modelPath = pathsConcat(experimentDirectory_, "model_last.bin");

  // flashlight optim mode
  auto flOptimLevel = FLAGS_amp_optim_mode.empty()
      ? fl::OptimLevel::DEFAULT
      : fl::OptimMode::toOptimLevel(FLAGS_amp_optim_mode);
  fl::OptimMode::get().setOptimLevel(flOptimLevel);
  if (FLAGS_amp_use_mixed_precision) {
    // Only set the optim mode to O1 if it was left empty
    FL_LOG_MASTER(INFO)
        << "Mixed precision training enabled. Will perform loss scaling.";
    if (FLAGS_amp_optim_mode.empty()) {
      FL_LOG_MASTER(INFO) << "Mixed precision training enabled with no "
                             "optim mode specified - setting optim mode to O1.";
      fl::OptimMode::get().setOptimLevel(fl::OptimLevel::O1);
    }
  }

  createSpecAugmentation();
  auto curAsrTrainSet = fl::app::asr::loadPrefetchDataset(
      asrTrainDataset_,
      FLAGS_data_prefetch_threads,
      true /* shuffle */,
      asrEpoch_ /* seed */);
  af::sync();

  if (batchIdx_ < FLAGS_decoder_tr_pretrainWindow) {
    FL_LOG_MASTER(INFO) << "ASR window training opening";
  }
  while (batchIdx_ < FLAGS_train_total_updates) {
    if (batchIdx_ == FLAGS_decoder_tr_pretrainWindow) {
      lmCriterion_->clearWindow();
      FL_LOG_MASTER(INFO) << "ASR window training closed";
    }

    // Advance Epoch
    if (batchIdx_ && batchIdx_ % asrTrainDataset_->size() == 0) {
      stopTimers();
      ++asrEpoch_;
      curAsrTrainSet = fl::app::asr::loadPrefetchDataset(
          asrTrainDataset_,
          FLAGS_data_prefetch_threads,
          true /* shuffle */,
          asrEpoch_ /* seed */);
      saveCheckpoint(modelPath);
      // logMemoryManagerStatus();
    }

    if (batchIdx_ && batchIdx_ % lmTrainDataset_->size() == 0) {
      stopTimers();
      ++lmEpoch_;
      lmTrainDataset_->shuffle(FLAGS_train_seed + lmEpoch_);
      saveCheckpoint(modelPath);
      // logMemoryManagerStatus();
    }

    // Run train
    runTimeMeter_.resume();
    asrBatchTimerMeter_.resume();
    trainAsrStep(curAsrTrainSet->get(batchIdx_ % asrTrainDataset_->size()));
    asrBatchTimerMeter_.stopAndIncUnit();

    lmBatchTimerMeter_.resume();
    if (batchIdx_ >= FLAGS_train_lm_start_updates) {
      trainLmStep(lmTrainDataset_->get(batchIdx_));
    }
    lmBatchTimerMeter_.stopAndIncUnit();
    ++batchIdx_;

    // Run evaluation and save best checkpoint
    if (FLAGS_train_report_updates &&
        batchIdx_ % FLAGS_train_report_updates == 0) {
      runEvaluation();
    }

    // Force saving checkpoint every given interval
    if (FLAGS_train_save_updates && batchIdx_ % FLAGS_train_save_updates == 0) {
      stopTimers();
      saveCheckpoint(modelPath, "." + std::to_string(batchIdx_));
    }
  }
}

void Trainer::runEvaluation() {
  // Run evaluation and update meters
  stopTimers();
  evalStep();
  syncMeters();
  auto progress = getProgress();
  FL_LOG_MASTER(INFO) << progress;
  if (isMaster()) {
    logWriter_ << progress << "\n" << std::flush;
  }

  // Save best checkpoints
  auto modelPath = fl::lib::pathsConcat(experimentDirectory_, "model_last.bin");
  for (const auto& meter : asrValidStatsMeters_) {
    const auto& tag = meter.first;
    auto wer = meter.second.wrdEdit.value()[0];
    if (asrBestValidWer_.find(tag) == asrBestValidWer_.end() ||
        wer < asrBestValidWer_[tag]) {
      asrBestValidWer_[tag] = wer;
      saveCheckpoint(modelPath, "." + tag);
    }
  }
  for (const auto& meter : lmValidLossMeters_) {
    const auto& tag = meter.first;
    auto loss = meter.second.value()[0];
    if (lmBestValidLoss_.find(tag) == lmBestValidLoss_.end() ||
        loss < lmBestValidLoss_[tag]) {
      lmBestValidLoss_[tag] = loss;
      saveCheckpoint(modelPath, "." + tag);
    }
  }

  resetMeters();
}

void Trainer::trainAsrStep(const std::vector<af::array>& batch) {
  encoder_->train();
  lmCriterion_->train();
  setLr();

  // 1. Sample
  sampleTimerMeter_.resume();
  auto asrInput = fl::input(batch[fl::app::asr::kInputIdx]);
  auto asrTarget = fl::noGrad(batch[fl::app::asr::kTargetIdx]);
  auto asrInputSizes = batch[fl::app::asr::kDurationIdx];
  auto asrTargetSizes = batch[fl::app::asr::kTargetSizeIdx];
  auto asrSampleNames =
      fl::app::asr::readSampleIds(batch[fl::app::asr::kSampleIdx]);

  sampleTimerMeter_.stop();
  asrDataStatsMeter_.add(asrInputSizes, asrTargetSizes);
  auto T = asrTarget.dims(0);
  if (T > 600) {
    FL_LOG(INFO) << "skipping";
  }

  // 2. Forward
  bool retrySample = false;
  do {
    retrySample = false;
    fwdTimeMeter_.resume();
    if (FLAGS_specaug_start_update >= 0 &&
        batchIdx_ >= FLAGS_specaug_start_update) {
      asrInput = specAug_->forward({asrInput}).front();
    }
    auto asrOutput =
        forwardSequentialModuleWithPadMask(asrInput, encoder_, asrInputSizes);
    af::sync();

    critFwdTimeMeter_.resume();
    auto asrLoss =
        sum(lmCriterion_
                ->forward(
                    {asrOutput,
                     asrTarget,
                     fl::noGrad(asrInputSizes),
                     fl::noGrad(asrTargetSizes)})
                .front(),
            {0});
    af::sync();
    fwdTimeMeter_.stop();
    critFwdTimeMeter_.stop();

    // check and logging training stats
    if (af::anyTrue<bool>(af::isNaN(asrLoss.array())) ||
        af::anyTrue<bool>(af::isInf(asrLoss.array()))) {
      LOG(FATAL) << "asrLoss has NaN/Inf values. Samples - "
                 << lib::join(",", asrSampleNames);
    }

    if (hasher_(lib::join(",", asrSampleNames)) % 100 <=
        FLAGS_exp_pct_train_eval) {
      asrTrainStatsMeter_.loss.add(asrLoss.array());
      evalWer(asrOutput.array(), asrTarget.array(), asrTrainStatsMeter_);
    }

    auto loss = asrLoss / (fl::getWorldSize() * FLAGS_data_asr_batch_size);
    if (FLAGS_amp_use_mixed_precision) {
      loss = loss * scaleFactor_;
    }
    if (scaleCounter_ == 20) {
      LOG(FATAL) << "WTF";
    }

    // 3. Backward
    bwdTimeMeter_.resume();
    ecdOptimizer_->zeroGrad();
    critOptimizer_->zeroGrad();
    loss.backward();
    reduceGrads();

    for (auto& p : parameters_) {
      p.grad() = p.grad() / scaleFactor_;
      if (FLAGS_amp_use_mixed_precision) {
        if (af::anyTrue<bool>(af::isNaN(p.grad().array())) ||
            af::anyTrue<bool>(af::isInf(p.grad().array()))) {
          if (scaleFactor_ >= fl::kAmpMinimumScaleFactorValue) {
            scaleFactor_ = scaleFactor_ / 2.0f;
            FL_LOG(INFO) << "AMP: Scale factor decreased. New value:\t"
                         << scaleFactor_;
            retrySample = true;
            ++scaleCounter_;
          } else {
            FL_LOG(FATAL) << "Minimum loss scale reached: "
                          << fl::kAmpMinimumScaleFactorValue
                          << " with over/underflowing gradients. Lowering the "
                          << "learning rate, using gradient clipping, or "
                          << "increasing the batch size can help resolve "
                          << "loss explosion.";
          }
          break;
        }
      }
    }
    af::sync();
    bwdTimeMeter_.stop();
  } while (retrySample);
  scaleCounter_ = 1;

  // 4. Optimization
  optimTimeMeter_.resume();
  fl::clipGradNorm(parameters_, FLAGS_train_max_grad_norm);
  ecdOptimizer_->step();
  critOptimizer_->step();
  af::sync();
  optimTimeMeter_.stop();

  // 5. update scale factor
  if (FLAGS_amp_use_mixed_precision &&
      scaleFactor_ < FLAGS_amp_max_scale_factor) {
    if (scaleCounter_ % FLAGS_amp_scale_factor_update_interval == 0) {
      scaleFactor_ *= 2;
      FL_VLOG(2) << "AMP: Scale factor doubled. New value:\t" << scaleFactor_;
    } else {
      scaleFactor_ += 2;
      FL_VLOG(3) << "AMP: Scale factor incremented. New value\t"
                 << scaleFactor_;
    }
  }
}

void Trainer::trainLmStep(const std::vector<af::array>& batch) {
  lmCriterion_->train();
  setLr();

  // 1. Sample
  sampleTimerMeter_.resume();
  fl::Variable lmInput = fl::noGrad(batch[0]), lmTarget = fl::noGrad(batch[0]);
  af::array lmInputSizes = af::flat(af::sum(lmInput.array() != kPadIdx_, 0));
  sampleTimerMeter_.stopAndIncUnit();

  // 2. Forward
  bool retrySample = false;
  do {
    retrySample = false;
    fwdTimeMeter_.resume();
    critFwdTimeMeter_.resume();
    auto lmLoss =
        sum(lmCriterion_->forward({fl::Variable(), lmInput}).front(), {0});
    af::sync();
    fwdTimeMeter_.stopAndIncUnit();
    critFwdTimeMeter_.stopAndIncUnit();

    // check and logging training stats
    if (af::anyTrue<bool>(af::isNaN(lmLoss.array())) ||
        af::anyTrue<bool>(af::isInf(lmLoss.array()))) {
      LOG(FATAL) << "lmLoss has NaN/Inf values.";
    }
    auto numTokens = af::count<float>(lmTarget.array() != kPadIdx_);
    if (numTokens > 0) {
      auto weight = numTokens /
          (FLAGS_data_lm_tokens_per_sample * FLAGS_data_lm_batch_size);
      lmTrainLossMeter_.add(
          af::mean<float>(lmLoss.array()) / numTokens, weight);
      lmTokenCountMeter_.add(numTokens);
    }
    af::array numTokensArr = af::array(1, &numTokens);
    if (FLAGS_distributed_enable) {
      fl::allReduce(numTokensArr);
    }

    auto loss = lmLoss / fl::Variable(numTokensArr, false);
    if (FLAGS_amp_use_mixed_precision) {
      loss = loss * scaleFactor_;
    }
    if (scaleCounter_ == 20) {
      LOG(FATAL) << "WTF";
    }

    // 3. Backward
    bwdTimeMeter_.resume();
    critOptimizer_->zeroGrad();
    loss.backward();
    reduceGrads();

    for (auto& p : lmCriterion_->params()) {
      p.grad() = p.grad() / scaleFactor_;
      if (FLAGS_amp_use_mixed_precision) {
        if (af::anyTrue<bool>(af::isNaN(p.grad().array())) ||
            af::anyTrue<bool>(af::isInf(p.grad().array()))) {
          if (scaleFactor_ >= fl::kAmpMinimumScaleFactorValue) {
            scaleFactor_ = scaleFactor_ / 2.0f;
            FL_LOG(INFO) << "AMP: Scale factor decreased. New value:\t"
                         << scaleFactor_;
            retrySample = true;
            ++scaleCounter_;
          } else {
            FL_LOG(FATAL) << "Minimum loss scale reached: "
                          << fl::kAmpMinimumScaleFactorValue
                          << " with over/underflowing gradients. Lowering the "
                          << "learning rate, using gradient clipping, or "
                          << "increasing the batch size can help resolve "
                          << "loss explosion.";
          }
          break;
        }
      }
    }
    af::sync();
    bwdTimeMeter_.stopAndIncUnit();
  } while (retrySample);
  scaleCounter_ = 1;

  // 4. Optimization
  optimTimeMeter_.resume();
  fl::clipGradNorm(lmCriterion_->params(), FLAGS_train_max_grad_norm);
  critOptimizer_->step();
  af::sync();
  optimTimeMeter_.stopAndIncUnit();

  // 5. update scale factor
  if (FLAGS_amp_use_mixed_precision &&
      scaleFactor_ < FLAGS_amp_max_scale_factor) {
    if (scaleCounter_ % FLAGS_amp_scale_factor_update_interval == 0) {
      scaleFactor_ *= 2;
      FL_VLOG(2) << "AMP: Scale factor doubled. New value:\t" << scaleFactor_;
    } else {
      scaleFactor_ += 2;
      FL_VLOG(3) << "AMP: Scale factor incremented. New value\t"
                 << scaleFactor_;
    }
  }
}

void Trainer::evalStep() {
  encoder_->eval();
  // asrFrontEnd_->eval();
  // lmFrontEnd_->eval();
  // asrCriterionLinear_->eval();
  // asrCriterion_->eval();
  lmCriterion_->eval();

  // ASR
  if (asrTrainStatsMeter_.wrdEdit.errorRate()[0] < 80) {
    for (const auto& set : asrValidDatasets_) {
      const auto tag = set.first;
      const auto validDataset = set.second;
      auto& validMeter = asrValidStatsMeters_[tag];
      validMeter.tknEdit.reset();
      validMeter.wrdEdit.reset();
      validMeter.loss.reset();

      for (const auto& batch : *validDataset) {
        auto input = fl::input(batch[fl::app::asr::kInputIdx]);
        auto target = fl::noGrad(batch[fl::app::asr::kTargetIdx]);
        auto inputSizes = batch[fl::app::asr::kDurationIdx];
        auto targetSizes = batch[fl::app::asr::kTargetSizeIdx];

        // auto output =
        //     forwardSequentialModuleWithPadMask(input, asrFrontEnd_,
        //     inputSizes);
        auto output = ext::forwardSequentialModuleWithPadMask(
            input, encoder_, inputSizes);
        // output = asrCriterionLinear_->forward({output}).front().as(f32);
        // auto loss = asrCriterion_->forward({output, target}).front();
        auto loss = lmCriterion_
                        ->forward(
                            {output,
                             target,
                             fl::noGrad(inputSizes),
                             fl::noGrad(targetSizes)})
                        .front();

        validMeter.loss.add(af::sum<double>(loss.array()));
        evalWer(output.array(), target.array(), validMeter);
      }
    }
  }

  // LM
  for (const auto& set : lmValidDatasets_) {
    const auto tag = set.first;
    const auto validDataset = set.second;
    auto& validMeter = lmValidLossMeters_[tag];

    for (const auto& batch : *validDataset) {
      fl::Variable input = fl::noGrad(batch[0]), target = fl::noGrad(batch[0]);
      // fl::Variable input, target;
      // std::tie(input, target) = getInputAndTarget(batch);
      // af::array inputSizes = af::flat(af::sum(input.array() != kPadIdx_, 0));
      // auto output = lmFrontEnd_->forward({input}).front();
      // output = forwardSequentialModuleWithPadMask(
      //     output, encoder_, inputSizes, true, output.dims(1),
      //     output.dims(2));
      // if (!FLAGS_dictionary_wordlm) {
      //   if (FLAGS_loss_type != "ce") {
      //     throw std::runtime_error("FLAGS_loss_type != ce");
      //   }
      //   output = asrCriterionLinear_->forward({output}).front();
      //   output = logSoftmax(output, 0);
      // }
      auto loss = lmCriterion_->forward({fl::Variable(), input}).front();
      // auto loss = fl::constant(0, 1, f32, false);
      auto numTokens = af::count<int>(target.array() != kPadIdx_);
      if (numTokens > 0) {
        auto weight =
            numTokens /
            static_cast<double>(
                FLAGS_data_lm_tokens_per_sample * FLAGS_data_lm_batch_size);
        validMeter.add(af::mean<double>(loss.array()) / numTokens, weight);
      }
    }
  }
}

/* ============= Initializers ============= */
void Trainer::initTrain() {
  FL_LOG_MASTER(INFO) << "Creating a fresh model";

  scaleFactor_ = FLAGS_amp_use_mixed_precision ? FLAGS_amp_scale_factor : 1.;
  scaleCounter_ = 1;

  createDictionary();
  createDatasets();
  createNetwork();
  createCriterion();
  createOptimizer();
}

void Trainer::initContinue() {
  auto checkPoint = pathsConcat(experimentDirectory_, "model_last.bin");
  if (!fileExists(checkPoint)) {
    throw std::invalid_argument(
        "Checkpoint doesn't exist to continue training: " + checkPoint);
  }
  FL_LOG_MASTER(INFO) << "Continue training from file: " << checkPoint;
  fl::ext::Serializer::load(
      checkPoint,
      version_,
      encoder_,
      // asrFrontEnd_,
      // lmFrontEnd_,
      // asrCriterionLinear_,
      // asrCriterion_,
      lmCriterion_,
      // optimizer_,
      ecdOptimizer_,
      critOptimizer_,
      asrEpoch_,
      lmEpoch_,
      batchIdx_,
      scaleFactor_,
      scaleCounter_,
      gflagsStr_,
      asrBestValidWer_,
      lmBestValidLoss_);

  // overwrite flags using the ones from command line
  gflags::ReadFlagsFromString(gflagsStr_, gflags::GetArgv0(), true);

  createDictionary();
  createDatasets();
  // the network, criterion and optimizer will be reused
}

void Trainer::initFork() {
  if (!fileExists(FLAGS_exp_init_model_path)) {
    throw std::invalid_argument(
        "Checkpoint doesn't exist for finetuning: " +
        FLAGS_exp_init_model_path);
  }
  FL_LOG_MASTER(INFO) << "Fork training from file: "
                      << FLAGS_exp_init_model_path;

  std::shared_ptr<fl::FirstOrderOptimizer> dummyOptimizer;
  fl::ext::Serializer::load(
      FLAGS_exp_init_model_path,
      version_,
      encoder_,
      // asrFrontEnd_,
      // lmFrontEnd_,
      // asrCriterionLinear_,
      // asrCriterion_,
      lmCriterion_,
      // optimizer_,
      ecdOptimizer_,
      critOptimizer_,
      asrEpoch_,
      lmEpoch_,
      batchIdx_,
      scaleFactor_,
      scaleCounter_,
      gflagsStr_,
      asrBestValidWer_,
      lmBestValidLoss_);

  createDictionary();
  createDatasets();
  createOptimizer();
  // the network and criterion will be reused
}

void Trainer::initEval() {
  if (!fileExists(FLAGS_exp_init_model_path)) {
    throw std::invalid_argument(
        "Checkpoint doesn't exist for finetuning: " +
        FLAGS_exp_init_model_path);
  }
  FL_LOG_MASTER(INFO) << "Fork training from file: "
                      << FLAGS_exp_init_model_path;

  std::shared_ptr<fl::FirstOrderOptimizer> dummyOptimizer;
  fl::ext::Serializer::load(
      FLAGS_exp_init_model_path,
      version_,
      encoder_,
      // asrFrontEnd_,
      // lmFrontEnd_,
      // asrCriterionLinear_,
      // asrCriterion_,
      lmCriterion_,
      // optimizer_,
      ecdOptimizer_,
      critOptimizer_,
      asrEpoch_,
      lmEpoch_,
      batchIdx_,
      scaleFactor_,
      scaleCounter_,
      gflagsStr_,
      asrBestValidWer_,
      lmBestValidLoss_);

  createDictionary();
  createDatasets();
  // createOptimizer();
  // the network and criterion will be reused
}

void Trainer::createDictionary() {
  if (FLAGS_dictionary_tokens.empty() ||
      !fl::lib::fileExists(FLAGS_dictionary_tokens)) {
    throw std::runtime_error(
        "Invalid dictionary filepath specified with "
        "--tokensdir and --tokens: \"" +
        FLAGS_dictionary_tokens + "\"");
  }

  // Token dictionary
  tokenDictionary_ = fl::lib::text::Dictionary(FLAGS_dictionary_tokens);
  // tokenDictionary_.addEntry(fl::app::asr::kBlankToken);
  numClasses_ = tokenDictionary_.indexSize();
  FL_LOG_MASTER(INFO) << "[Number of tokens] " << numClasses_;

  // Word dictionary
  if (FLAGS_dictionary.empty()) {
    throw std::invalid_argument("Lexicon is empty");
  }
  lexicon_ =
      fl::lib::text::loadWords(FLAGS_dictionary, FLAGS_dictionary_max_size);
  wordDictionary_ = fl::lib::text::createWordDict(lexicon_);
  // wordDictionary_.addEntry(fl::lib::text::kPadToken);
  // wordDictionary_.addEntry(fl::lib::text::kEosToken);
  // wordDictionary_.addEntry(fl::lib::text::kMaskToken);
  // wordDictionary_.addEntry(fl::lib::text::kUnkToken);
  // for (const auto& it : lexicon_) {
  //   if (it.first == fl::lib::text::kUnkToken) {
  //     continue;
  //   }
  //   wordDictionary_.addEntry(it.first);
  // }
  FL_LOG_MASTER(INFO) << "[Number of words] " << wordDictionary_.indexSize();

  // if (!wordDictionary_.isContiguous()) {
  //   throw std::runtime_error("Invalid wordDictionary_ format - not
  //   contiguous");
  // }
  // kPadIdx_ = wordDictionary_.getIndex(fl::lib::text::kPadToken);
  // kEosIdx_ = wordDictionary_.getIndex(fl::lib::text::kEosToken);
  // kUnkIdx_ = wordDictionary_.getIndex(fl::lib::text::kUnkToken);
  // kMaskIdx_ = wordDictionary_.getIndex(fl::lib::text::kMaskToken);
  // wordDictionary_.setDefaultIndex(kUnkIdx_);

  // LM dictionary
  lmDictionary_ = fl::lib::text::Dictionary();
  lmDictionary_.addEntry(fl::lib::text::kPadToken);
  lmDictionary_.addEntry(fl::lib::text::kEosToken);
  lmDictionary_.addEntry(fl::lib::text::kMaskToken);
  lmDictionary_.addEntry(fl::lib::text::kUnkToken);
  if (FLAGS_dictionary_wordlm) {
    for (int i = 0; i < wordDictionary_.indexSize(); i++) {
      auto word = wordDictionary_.getEntry(i);
      if (word == fl::lib::text::kUnkToken) {
        continue;
      }
      lmDictionary_.addEntry(word);
    }
  } else {
    for (int i = 0; i < tokenDictionary_.indexSize(); i++) {
      auto word = tokenDictionary_.getEntry(i);
      lmDictionary_.addEntry(word);
    }
    numClasses_ = lmDictionary_.indexSize();
    tokenDictionary_ = lmDictionary_;
  }
  kPadIdx_ = lmDictionary_.getIndex(fl::lib::text::kPadToken);
  kEosIdx_ = lmDictionary_.getIndex(fl::lib::text::kEosToken);
  kUnkIdx_ = lmDictionary_.getIndex(fl::lib::text::kUnkToken);
  kMaskIdx_ = lmDictionary_.getIndex(fl::lib::text::kMaskToken);
  lmDictionary_.setDefaultIndex(kUnkIdx_);
  FL_LOG_MASTER(INFO) << "[Number of LM words] " << lmDictionary_.indexSize();
}

void Trainer::createDatasets() {
  // ASR
  fl::lib::audio::FeatureParams featParams(
      FLAGS_feat_samplerate,
      FLAGS_feat_framesizems,
      FLAGS_feat_framestridems,
      FLAGS_feat_filterbanks,
      FLAGS_feat_lowfreqfilterbank,
      FLAGS_feat_highfreqfilterbank,
      FLAGS_feat_mfcccoeffs,
      fl::app::asr::kLifterParam /* lifterparam */,
      FLAGS_feat_devwin /* delta window */,
      FLAGS_feat_devwin /* delta-delta window */);
  featParams.useEnergy = false;
  featParams.usePower = false;
  featParams.zeroMeanFrame = false;
  numFeatures_ = -1;
  using fl::app::asr::FeatureType;
  FeatureType featType = FeatureType::NONE;
  if (FLAGS_feat_pow) {
    featType = FeatureType::POW_SPECTRUM;
    numFeatures_ = featParams.powSpecFeatSz();
  } else if (FLAGS_feat_mfsc) {
    featType = FeatureType::MFSC;
    numFeatures_ = featParams.mfscFeatSz();
  } else if (FLAGS_feat_mfcc) {
    featType = FeatureType::MFCC;
    numFeatures_ = featParams.mfccFeatSz();
  }
  fl::app::asr::TargetGenerationConfig targetGenConfig(
      FLAGS_data_asr_wordseparator,
      FLAGS_data_asr_sampletarget,
      "s2s",
      FLAGS_data_asr_surround,
      FLAGS_data_asr_eostoken,
      FLAGS_data_asr_replabel,
      true /* skip unk */,
      FLAGS_data_asr_usewordpiece /* fallback2LetterWordSepLeft */,
      !FLAGS_data_asr_usewordpiece /* fallback2LetterWordSepLeft */);

  const auto sfxConf = (FLAGS_ssfx_config.empty())
      ? std::vector<fl::app::asr::sfx::SoundEffectConfig>()
      : fl::app::asr::sfx::readSoundEffectConfigFile(FLAGS_ssfx_config);

  auto inputTransform = inputFeatures(
      featParams,
      featType,
      {FLAGS_norm_localnrmlleftctx, FLAGS_norm_localnrmlrightctx},
      sfxConf);
  auto targetTransform =
      targetFeatures(tokenDictionary_, lexicon_, targetGenConfig);
  auto wordTransform = fl::app::asr::wordFeatures(wordDictionary_);
  int targetpadVal = lmDictionary_.getIndex(fl::lib::text::kPadToken);
  int wordpadVal = fl::app::asr::kTargetPadValue;

  std::vector<std::string> trainSplits =
      fl::lib::split(",", FLAGS_data_asr_train, true);
  asrTrainDataset_ = fl::app::asr::createDataset(
      trainSplits,
      FLAGS_data_asr_dir,
      FLAGS_data_asr_batch_size,
      inputTransform,
      targetTransform,
      wordTransform,
      std::make_tuple(0, targetpadVal, wordpadVal),
      fl::getWorldRank(),
      fl::getWorldSize());
  FL_LOG_MASTER(INFO) << "[ASR train dataset] Loaded "
                      << asrTrainDataset_->size() * fl::getWorldSize()
                      << " samples";

  auto validSets =
      fl::lib::split(',', fl::lib::trim(FLAGS_data_asr_valid), true);
  for (const auto& s : validSets) {
    std::string tag, path;
    std::tie(tag, path) = parseDatasetName(s);

    asrValidDatasets_[tag] = fl::app::asr::createDataset(
        {path},
        FLAGS_data_asr_dir,
        1 /* FLAGS_data_asr_batch_size */,
        inputTransform,
        targetTransform,
        wordTransform,
        std::make_tuple(0, targetpadVal, wordpadVal),
        fl::getWorldRank(),
        fl::getWorldSize());
    asrValidStatsMeters_[tag] = fl::app::asr::DatasetMeters();
    FL_LOG_MASTER(INFO) << "[ASR valid dataset: " << tag << "] Loaded "
                        << asrValidDatasets_[tag]->size() * fl::getWorldSize()
                        << " samples";
  }

  // LM
  fl::lib::text::Tokenizer tokenizer;
  fl::lib::text::PartialFileReader partialFileReader(
      fl::getWorldRank(), fl::getWorldSize());
  lmTrainDataset_ = std::make_shared<fl::app::lm::TextDataset>(
      FLAGS_data_lm_dir,
      FLAGS_data_lm_train,
      partialFileReader,
      tokenizer,
      lmDictionary_,
      FLAGS_data_lm_tokens_per_sample,
      FLAGS_data_lm_batch_size,
      FLAGS_data_lm_sample_break_mode,
      true);
  FL_LOG_MASTER(INFO) << "[LM train dataset] Loaded "
                      << lmTrainDataset_->size() * fl::getWorldSize()
                      << " samples";

  validSets = fl::lib::split(',', fl::lib::trim(FLAGS_data_lm_valid), true);
  for (const auto& s : validSets) {
    std::string tag, path;
    std::tie(tag, path) = parseDatasetName(s);
    lmValidDatasets_[tag] = std::make_shared<fl::app::lm::TextDataset>(
        FLAGS_data_lm_dir,
        path,
        partialFileReader,
        tokenizer,
        lmDictionary_,
        FLAGS_data_lm_tokens_per_sample,
        FLAGS_data_lm_batch_size,
        "eos",
        FLAGS_data_lm_use_dynamic_batching);
    FL_LOG_MASTER(INFO) << "[LM valid dataset: " << tag << "] Loaded "
                        << lmValidDatasets_[tag]->size() * fl::getWorldSize()
                        << " samples";
  }
}

void Trainer::createNetwork() {
  // Encoder
  auto archfile =
      fl::lib::pathsConcat(FLAGS_train_arch_dir, FLAGS_train_arch_file);
  encoder_ =
      fl::ext::buildSequentialModule(archfile, numFeatures_, numClasses_);
  FL_LOG_MASTER(INFO) << "[Encoder] " << encoder_->prettyString();
  FL_LOG_MASTER(INFO) << "[Encoder Params: " << numTotalParams(encoder_) << "]";

  // Front-ends
  // archfile = fl::lib::pathsConcat(
  //     FLAGS_train_arch_dir, FLAGS_train_asr_frontend_arch_file);
  // asrFrontEnd_ =
  //     fl::ext::buildSequentialModule(archfile, numFeatures_, numClasses_);
  // FL_LOG_MASTER(INFO) << "[ASR front-end] " << asrFrontEnd_->prettyString();
  // FL_LOG_MASTER(INFO) << "[ASR front-end Params: "
  //                     << numTotalParams(asrFrontEnd_) << "]";

  // archfile = fl::lib::pathsConcat(
  //     FLAGS_train_arch_dir, FLAGS_train_lm_frontend_arch_file);
  // lmFrontEnd_ =
  //     fl::ext::buildSequentialModule(archfile, 0, lmDictionary_.entrySize());
  // FL_LOG_MASTER(INFO) << "[LM front-end] " << lmFrontEnd_->prettyString();
  // FL_LOG_MASTER(INFO) << "[LM front-end Params: " <<
  // numTotalParams(lmFrontEnd_)
  //                     << "]";
}

void Trainer::createCriterion() {
  // ASR
  // asrCriterionLinear_ =
  //     std::make_shared<Linear>(FLAGS_loss_adsm_input_size, numClasses_,
  //     false);
  // auto scalemode =
  //     fl::app::asr::getCriterionScaleMode(FLAGS_norm_onorm,
  //     FLAGS_norm_sqnorm);
  // asrCriterion_ = std::make_shared<fl::app::asr::CTCLoss>(scalemode);
  // FL_LOG_MASTER(INFO) << "[ASR Criterion] "
  //                     << asrCriterionLinear_->prettyString() << " + "
  //                     << asrCriterion_->prettyString();

  // LM
  // if (FLAGS_loss_type == "adsm") {
  //   if (lmDictionary_.entrySize() == 0) {
  //     throw std::runtime_error(
  //         "Dictionary is empty, number of classes is zero");
  //   }
  //   auto softmax = std::make_shared<fl::AdaptiveSoftMax>(
  //       FLAGS_loss_adsm_input_size, parseCutoffs(lmDictionary_.entrySize()));
  //   lmCriterion_ = std::make_shared<fl::AdaptiveSoftMaxLoss>(
  //       softmax, fl::ReduceMode::SUM, kPadIdx_);
  // } else if (FLAGS_loss_type == "ce") {
  //   lmCriterion_ = std::make_shared<fl::CategoricalCrossEntropy>(
  //       fl::ReduceMode::SUM, kPadIdx_);
  // } else {
  //   throw std::runtime_error(
  //       "Criterion is not supported, check 'loss_type' flag possible
  //       values");
  // }

  lmCriterion_ = std::make_shared<fl::app::asr::TransformerCriterion>(
      lmDictionary_.entrySize(),
      FLAGS_decoder_tr_encoderdim,
      lmDictionary_.getIndex(fl::app::asr::kEosToken),
      lmDictionary_.getIndex(fl::lib::text::kPadToken),
      100, // FLAGS_maxdecoderoutputlen,
      FLAGS_decoder_tr_layers,
      fl::app::asr::createAttention(),
      fl::app::asr::createAttentionWindow(),
      true, // FLAGS_trainWithWindow,
      0.05, // FLAGS_labelsmooth,
      99, // FLAGS_pctteacherforcing,
      FLAGS_decoder_tr_dropout,
      FLAGS_decoder_tr_layerdrop);

  FL_LOG_MASTER(INFO) << "[LM Criterion] " << lmCriterion_->prettyString();
}

void Trainer::collectParameters() {
  parameters_ = encoder_->params();
  // const auto& asrFrontEndParams = asrFrontEnd_->params();
  // parameters_.insert(
  //     parameters_.end(), asrFrontEndParams.begin(), asrFrontEndParams.end());
  // // const auto& lmFrontEndParams = lmFrontEnd_->params();
  // // parameters_.insert(
  // //     parameters_.end(), lmFrontEndParams.begin(),
  // lmFrontEndParams.end());
  // // const auto& asrCriterionLinearParams = asrCriterionLinear_->params();
  // // parameters_.insert(
  // //     parameters_.end(),
  // //     asrCriterionLinearParams.begin(),
  // //     asrCriterionLinearParams.end());
  // // const auto& asrCriterionParams = asrCriterion_->params();
  // // parameters_.insert(
  // //     parameters_.end(), asrCriterionParams.begin(),
  // asrCriterionParams.end());
  const auto& lmCriterionParams = lmCriterion_->params();
  parameters_.insert(
      parameters_.end(), lmCriterionParams.begin(), lmCriterionParams.end());
}

void Trainer::createOptimizer() {
  collectParameters();
  if (FLAGS_train_optimizer == "nag") {
    ecdOptimizer_ = std::make_shared<fl::NAGOptimizer>(
        encoder_->params(),
        FLAGS_train_lr,
        FLAGS_train_momentum,
        FLAGS_train_weight_decay);
    critOptimizer_ = std::make_shared<fl::NAGOptimizer>(
        lmCriterion_->params(),
        FLAGS_train_lr,
        FLAGS_train_momentum,
        FLAGS_train_weight_decay);
  } else if (FLAGS_train_optimizer == "sgd") {
    ecdOptimizer_ = std::make_shared<fl::SGDOptimizer>(
        encoder_->params(),
        FLAGS_train_lr,
        FLAGS_train_momentum,
        FLAGS_train_weight_decay,
        false);
    critOptimizer_ = std::make_shared<fl::SGDOptimizer>(
        lmCriterion_->params(),
        FLAGS_train_lr,
        FLAGS_train_momentum,
        FLAGS_train_weight_decay,
        false);
  } else if (FLAGS_train_optimizer == "adagrad") {
    ecdOptimizer_ = std::make_shared<fl::AdagradOptimizer>(
        encoder_->params(), FLAGS_train_lr, 1e-8, FLAGS_train_weight_decay);
    critOptimizer_ = std::make_shared<fl::AdagradOptimizer>(
        lmCriterion_->params(), FLAGS_train_lr, 1e-8, FLAGS_train_weight_decay);
  } else {
    throw std::runtime_error(
        "Optimizer is not supported, check 'train_optimizer' flag possible values");
  }
}

void Trainer::createSpecAugmentation() {
  if (FLAGS_specaug_start_update >= 0) {
    if (!(FLAGS_feat_pow || FLAGS_feat_mfsc || FLAGS_feat_mfcc)) {
      specAug_ = std::make_shared<fl::RawWavSpecAugment>(
          FLAGS_feat_filterbanks,
          FLAGS_specaug_fmaskf,
          FLAGS_specaug_fmaskn,
          FLAGS_specaug_tmaskt,
          FLAGS_specaug_tmaskp,
          FLAGS_specaug_tmaskn,
          FLAGS_feat_filterbanks,
          FLAGS_feat_lowfreqfilterbank,
          FLAGS_feat_highfreqfilterbank,
          FLAGS_feat_samplerate);
    } else {
      specAug_ = std::make_shared<fl::SpecAugment>(
          FLAGS_feat_filterbanks,
          FLAGS_specaug_fmaskf,
          FLAGS_specaug_fmaskn,
          FLAGS_specaug_tmaskt,
          FLAGS_specaug_tmaskp,
          FLAGS_specaug_tmaskn);
    }
  }
}

/* ============= Stateful training helpers ============= */
std::pair<fl::Variable, fl::Variable> Trainer::getInputAndTarget(
    const std::vector<af::array>& sample) const {
  // sample.size() == 1
  // sample[0] has size T x B
  fl::Variable input, target;
  auto T = sample[0].dims(0);

  if (FLAGS_train_task == "mask") {
    // TODO: need cleaning + correctness checking

    // do masking of input and target
    af::array randMatrix = af::randu(sample[0].dims());
    af::array randMatrixSorted, randMatrixSortedIndices;
    // create random permutation
    af::sort(randMatrixSorted, randMatrixSortedIndices, randMatrix, 0);
    randMatrixSortedIndices = af::flat(randMatrixSortedIndices);

    af::array inputMasked = af::flat(sample[0]);
    // set total mask
    af::array totalMask = randMatrixSortedIndices < FLAGS_mask_prob * T;
    // set min length of the masked tokens
    int nTotalMask =
        std::max(int(FLAGS_mask_prob * T), (int)FLAGS_mask_min_length);
    if (FLAGS_mask_min_length > 0) {
      totalMask =
          totalMask + (randMatrixSortedIndices < FLAGS_mask_min_length) > 0;
    }
    af::array notMasked = (1 - totalMask).as(b8);
    af::array woMaskTokenMask = randMatrixSortedIndices <
        (FLAGS_mask_rand_token_prob + FLAGS_mask_same_token_prob) * nTotalMask;
    af::array randMask =
        randMatrixSortedIndices < FLAGS_mask_rand_token_prob * nTotalMask;

    inputMasked(totalMask) = kMaskIdx_;
    inputMasked(woMaskTokenMask) = af::flat(sample[0])(woMaskTokenMask);
    if (af::sum(randMask).scalar<unsigned int>() > 0) {
      // exclude 4 special tokens from the consideration: pad, eos, unk and
      // mask
      int nSpecialTokens = 4;
      inputMasked(randMask) =
          (af::randu(af::sum(randMask).scalar<unsigned int>()) *
               (lmDictionary_.entrySize() - nSpecialTokens - 1) +
           nSpecialTokens)
              .as(s32);
    }
    // fix position where it was pad index to be pad
    inputMasked(af::flat(sample[0] == kPadIdx_)) = kPadIdx_;
    inputMasked = af::moddims(inputMasked, sample[0].dims());
    input = fl::Variable(inputMasked, false);
    auto targetMasked = af::flat(sample[0]);
    targetMasked(notMasked) = kPadIdx_;
    targetMasked = af::moddims(targetMasked, sample[0].dims());
    target = fl::Variable(targetMasked, false);
  } else if (FLAGS_train_task == "autoreg") {
    input = fl::Variable(sample[0](af::seq(0, T - 2), af::span), false);
    target = fl::Variable(sample[0](af::seq(1, T - 1), af::span), false);
  } else {
    throw std::invalid_argument(
        "Not supported train_task: " + FLAGS_train_task);
  }
  return std::make_pair(input, target);
}

void Trainer::setLr() {
  if (batchIdx_ < FLAGS_train_warmup_updates) {
    // warmup stage
    lr_ = FLAGS_train_warmup_init_lr +
        (FLAGS_train_lr - FLAGS_train_warmup_init_lr) * batchIdx_ /
            (double(FLAGS_train_warmup_updates));
  } else {
    if (FLAGS_train_lr_schedule == "fixed") {
      // after warmup stage + fixed policy
      lr_ = FLAGS_train_lr;
    } else if (FLAGS_train_lr_schedule == "invsqrt") {
      // after warmup stage + invsqrt policy
      if (FLAGS_train_warmup_updates > 0) {
        lr_ = FLAGS_train_lr * std::sqrt(FLAGS_train_warmup_updates) /
            std::sqrt(batchIdx_);
      } else {
        lr_ = FLAGS_train_lr / std::sqrt(batchIdx_ + 1);
      }
    } else {
      throw std::runtime_error(
          "LR schedule is not supported, check train_lr_schedule flag possible values");
    }
  }
  // optimizer_->setLr(lr_);
  ecdOptimizer_->setLr(lr_);
  critOptimizer_->setLr(lr_);
}

void Trainer::reduceGrads() {
  collectParameters();
  if (reducer_) {
    // for (auto& p : parameters_) {
    //   if (!p.isGradAvailable()) {
    //     p.addGrad(fl::constant(0.0, p.dims(), p.type(), false));
    //   }
    //   auto& grad = p.grad().array();
    //   p.grad().array() = grad;
    //   reducer_->add(p.grad());
    // }
    reducer_->finalize();
  }
}

void Trainer::evalWer(
    const af::array& output,
    const af::array& target,
    fl::app::asr::DatasetMeters& meter) {
  auto batchSize = output.dims(2);

  for (int b = 0; b < batchSize; ++b) {
    auto viterbiPath = fl::ext::afToVector<int>(
        lmCriterion_->viterbiPath(output(af::span, af::span, b)));
    // asrCriterion_->viterbiPath(output(af::span, af::span, b)));
    auto rawTarget = fl::ext::afToVector<int>(target(af::span, b));

    // Remove `-1`s appended to the target for batching (if any)
    auto labellen =
        fl::app::asr::getTargetSize(rawTarget.data(), rawTarget.size());
    rawTarget.resize(labellen);

    // remap actual, predicted targets for evaluating edit distance error
    auto letterPrediction = fl::app::asr::tknPrediction2Ltr(
        viterbiPath,
        tokenDictionary_,
        "s2s",
        FLAGS_data_asr_surround,
        FLAGS_data_asr_eostoken,
        FLAGS_data_asr_replabel,
        FLAGS_data_asr_usewordpiece,
        FLAGS_data_asr_wordseparator);
    auto letterTarget = fl::app::asr::tknTarget2Ltr(
        rawTarget,
        tokenDictionary_,
        "s2s",
        FLAGS_data_asr_surround,
        FLAGS_data_asr_eostoken,
        FLAGS_data_asr_replabel,
        FLAGS_data_asr_usewordpiece,
        FLAGS_data_asr_wordseparator);

    auto wordPrediction =
        fl::app::asr::tkn2Wrd(letterPrediction, FLAGS_data_asr_wordseparator);
    auto wordTarget =
        fl::app::asr::tkn2Wrd(letterTarget, FLAGS_data_asr_wordseparator);

    // std::cout << "\n|T|: ";
    // for (auto& word : wordTarget) {
    //   std::cout << word << " ";
    // }
    // std::cout << "\n|P|: ";
    // for (auto& word : wordPrediction) {
    //   std::cout << word << " ";
    // }
    // std::cout << "\n---";
    meter.tknEdit.add(letterPrediction, letterTarget);
    meter.wrdEdit.add(wordPrediction, wordTarget);
  }
}

/* ============= Stateless training helpers ============= */
void Trainer::initArrayFire() const {
  // Set arrayfire seed for reproducibility
  af::setSeed(FLAGS_train_seed);
}

std::vector<int> Trainer::parseCutoffs(int64_t nClasses) const {
  // parse cutoffs for adaptive softmax
  std::vector<int> cutoffs;
  auto tokens = lib::split(',', FLAGS_loss_adsm_cutoffs, true);
  for (const auto& token : tokens) {
    cutoffs.push_back(std::stoi(trim(token)));
  }
  cutoffs.push_back(nClasses);
  for (int i = 0; i + 1 < cutoffs.size(); ++i) {
    if (cutoffs[i] >= cutoffs[i + 1]) {
      throw std::invalid_argument(
          "Cutoffs for adaptive softmax must be strictly ascending, please fix the loss_adsm_cutoffs flag");
    }
  }
  return cutoffs;
}

std::pair<std::string, std::string> Trainer::parseDatasetName(
    const std::string& name) const {
  // assume the format is tag:filepath
  std::string tag, path;
  auto parts = fl::lib::splitOnAnyOf(":", name);
  if (parts.size() == 1) {
    tag = parts[0];
    path = parts[0];
  } else if (parts.size() == 2) {
    tag = parts[0];
    path = parts[1];
  } else {
    LOG(FATAL) << "invalid valid set: " << name;
  }
  return std::make_pair(tag, path);
}

bool Trainer::isMaster() const {
  return fl::getWorldRank() == 0;
}

void Trainer::checkArgs() const {
  if (version_ != FL_APP_JOINT_VERSION) {
    FL_LOG_MASTER(INFO) << "Model version (" << version_
                        << ") does not match FL_APP_LM_VERSION ("
                        << FL_APP_LM_VERSION << ")";
  }

  if (FLAGS_dictionary_max_size == 0) {
    throw std::invalid_argument(
        "'--dictionary_max_size' should be positive or -1");
  }
}

/* ============= Meter helpers ============= */
void Trainer::resetMeters() {
  asrTrainStatsMeter_.tknEdit.reset();
  asrTrainStatsMeter_.wrdEdit.reset();
  asrTrainStatsMeter_.loss.reset();
  asrDataStatsMeter_.reset();
  for (auto& meter : asrValidStatsMeters_) {
    auto& validMeter = meter.second;
    validMeter.tknEdit.reset();
    validMeter.wrdEdit.reset();
    validMeter.loss.reset();
  }

  lmTrainLossMeter_.reset();
  for (auto& meter : lmValidLossMeters_) {
    meter.second.reset();
  }

  runTimeMeter_.reset();
  asrBatchTimerMeter_.reset();
  lmBatchTimerMeter_.reset();
  sampleTimerMeter_.reset();
  fwdTimeMeter_.reset();
  critFwdTimeMeter_.reset();
  bwdTimeMeter_.reset();
  optimTimeMeter_.reset();
}

void Trainer::syncMeters() {
  syncMeter(asrTrainStatsMeter_.tknEdit);
  syncMeter(asrTrainStatsMeter_.wrdEdit);
  syncMeter(asrTrainStatsMeter_.loss);
  for (auto& meter : asrValidStatsMeters_) {
    auto& validMeter = meter.second;
    syncMeter(validMeter.tknEdit);
    syncMeter(validMeter.wrdEdit);
    syncMeter(validMeter.loss);
  }
  syncMeter(asrDataStatsMeter_);

  syncMeter(lmTrainLossMeter_);
  for (auto& meter : lmValidLossMeters_) {
    syncMeter(meter.second);
  }

  syncMeter(runTimeMeter_);
  syncMeter(asrBatchTimerMeter_);
  syncMeter(lmBatchTimerMeter_);
  syncMeter(sampleTimerMeter_);
  syncMeter(fwdTimeMeter_);
  syncMeter(critFwdTimeMeter_);
  syncMeter(bwdTimeMeter_);
  syncMeter(optimTimeMeter_);
  syncMeter(lmTokenCountMeter_);
}

void Trainer::stopTimers() {
  runTimeMeter_.stop();
  asrBatchTimerMeter_.stop();
  lmBatchTimerMeter_.stop();
  sampleTimerMeter_.stop();
  fwdTimeMeter_.stop();
  critFwdTimeMeter_.stop();
  bwdTimeMeter_.stop();
  optimTimeMeter_.stop();
}

/* ============= Logging helpers ============= */
void Trainer::saveCheckpoint(const std::string& path, const std::string& suffix)
    const {
  if (!isMaster()) {
    return;
  }

  // FL_LOG_MASTER(INFO) << "Saving model checkpoint (asr-epoch=" << asrEpoch_
  //                     << ", lm-epoch=" << lmEpoch_ << ", batch=" << batchIdx_
  //                     << ") to: " << path;
  Serializer::save(
      path,
      FL_APP_JOINT_VERSION,
      encoder_,
      // asrFrontEnd_,
      // lmFrontEnd_,
      // asrCriterionLinear_,
      // asrCriterion_,
      lmCriterion_,
      // optimizer_,
      ecdOptimizer_,
      critOptimizer_,
      asrEpoch_,
      lmEpoch_,
      batchIdx_,
      scaleFactor_,
      scaleCounter_,
      gflagsStr_,
      asrBestValidWer_,
      lmBestValidLoss_);

  if (!suffix.empty()) {
    Serializer::save(
        path + suffix,
        FL_APP_JOINT_VERSION,
        encoder_,
        // asrFrontEnd_,
        // lmFrontEnd_,
        // asrCriterionLinear_,
        // asrCriterion_,
        lmCriterion_,
        // optimizer_,
        ecdOptimizer_,
        critOptimizer_,
        asrEpoch_,
        lmEpoch_,
        batchIdx_,
        scaleFactor_,
        scaleCounter_,
        gflagsStr_,
        asrBestValidWer_,
        lmBestValidLoss_);
  }
}

void Trainer::logMemoryManagerStatus() const {
  if (isMaster()) {
    auto* curMemMgr =
        fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
    if (curMemMgr) {
      curMemMgr->printInfo("Memory Manager Stats", 0 /* device id */);
    }
  }
}

std::string Trainer::getProgress() const {
  std::string status;
  auto insertItem = [&](std::string key, std::string val) {
    val = key + ": " + val;
    status = status + (status.empty() ? "" : " | ") + val;
  };

  using fl::lib::format;

  // Timer
  insertItem("timestamp", lib::getCurrentDate() + " " + lib::getCurrentTime());
  insertItem("asr-epoch", format("%8d", asrEpoch_));
  insertItem("lm-epoch", format("%8d", lmEpoch_));
  insertItem("nupdates", format("%12d", batchIdx_));
  insertItem("lr", format("%4.6lf", lr_));
  insertItem("lrcriterion", format("%4.6lf", lr_));

  int runTime = runTimeMeter_.value();
  insertItem(
      "runtime",
      format(
          "%02d:%02d:%02d",
          (runTime / 60 / 60),
          (runTime / 60) % 60,
          runTime % 60));
  insertItem("asr-bch(ms)", format("%.2f", asrBatchTimerMeter_.value() * 1000));
  insertItem("lm-bch(ms)", format("%.2f", lmBatchTimerMeter_.value() * 1000));
  insertItem("smp(ms)", format("%.2f", sampleTimerMeter_.value() * 1000));
  insertItem("fwd(ms)", format("%.2f", fwdTimeMeter_.value() * 1000));
  insertItem("crit-fwd(ms)", format("%.2f", critFwdTimeMeter_.value() * 1000));
  insertItem("bwd(ms)", format("%.2f", bwdTimeMeter_.value() * 1000));
  insertItem("optim(ms)", format("%.2f", optimTimeMeter_.value() * 1000));

  // ASR
  insertItem("loss", format("%10.5f", asrTrainStatsMeter_.loss.value()[0]));
  insertItem(
      "train-TER", format("%5.2f", asrTrainStatsMeter_.tknEdit.errorRate()[0]));
  insertItem(
      "train-WER", format("%5.2f", asrTrainStatsMeter_.wrdEdit.errorRate()[0]));

  for (const auto& meter : asrValidStatsMeters_) {
    const auto tag = meter.first;
    const auto& validMeter = meter.second;
    insertItem(tag + "-loss", format("%10.5f", validMeter.loss.value()[0]));
    insertItem(
        tag + "-TER", format("%5.2f", validMeter.tknEdit.errorRate()[0]));
    insertItem(
        tag + "-WER", format("%5.2f", validMeter.wrdEdit.errorRate()[0]));
  }

  auto stats = asrDataStatsMeter_.value();
  auto numsamples = std::max<int64_t>(stats[4], 1);
  auto isztotal = stats[0];
  auto tsztotal = stats[1];
  auto tszmax = stats[3];
  insertItem("avg-isz", format("%03d", isztotal / numsamples));
  insertItem("avg-tsz", format("%03d", tsztotal / numsamples));
  insertItem("max-tsz", format("%03d", tszmax));

  double audioProcSec = isztotal * FLAGS_data_asr_batch_size;
  if (FLAGS_feat_pow || FLAGS_feat_mfcc || FLAGS_feat_mfsc) {
    audioProcSec = audioProcSec * FLAGS_feat_framestridems / 1000.0;
  } else {
    audioProcSec /= FLAGS_feat_samplerate;
  }
  auto worldSize = fl::getWorldSize();
  double timeTakenSec = asrBatchTimerMeter_.value() * numsamples / worldSize;

  insertItem("hrs", format("%7.2f", audioProcSec / 3600.0));
  insertItem(
      "thrpt(sec/sec)",
      timeTakenSec > 0.0 ? format("%.2f", audioProcSec / timeTakenSec) : "n/a");

  // LM
  double loss = lmTrainLossMeter_.value()[0];
  insertItem("lm-train-loss", format("%.2f", loss));
  insertItem("lm-train-ppl", format("%.2f", std::exp(loss)));
  for (const auto& meter : lmValidLossMeters_) {
    const auto tag = meter.first;
    const auto loss = meter.second.value()[0];
    insertItem("lm-" + tag + "-loss", format("%.2f", loss));
    insertItem("lm-" + tag + "-ppl", format("%.2f", std::exp(loss)));
  }
  insertItem(
      "lm-throughput",
      format(
          "%.2f",
          lmTokenCountMeter_.value()[0] * fl::getWorldSize() /
              lmBatchTimerMeter_.value()));
  return status;
}

#if 0
void Trainer::evalLM() {
  FL_LOG_MASTER(INFO) << "LM evaluation started.";
  asrFrontEnd_->eval();
  lmFrontEnd_->eval();
  encoder_->eval();
  asrCriterionLinear_->eval();
  asrCriterion_->eval();
  lmCriterion_->eval();

  int cnt = 0;
  for (const auto& set : lmValidDatasets_) {
    const auto tag = set.first;
    const auto validDataset = set.second;

    for (const auto& batch : *validDataset) {
      if (cnt++ == 100) {
        break;
      }
      fl::Variable input, target;
      std::tie(input, target) = getInputAndTarget(batch);
      af::array inputSizes = af::flat(af::sum(input.array() != kPadIdx_, 0));
      auto output = lmFrontEnd_->forward({input}).front();
      output = forwardSequentialModuleWithPadMask(
          output, encoder_, inputSizes, true, output.dims(1), output.dims(2));
      if (!FLAGS_dictionary_wordlm) {
        if (FLAGS_loss_type != "ce") {
          throw std::runtime_error("FLAGS_loss_type != ce");
        }
        output = asrCriterionLinear_->forward({output}).front();
        output = logSoftmax(output, 0);
      }

      for (int b = 0; b < output.dims(2); ++b) {
        auto viterbiPath = fl::ext::afToVector<int>(
            asrCriterion_->viterbiPath(output.array()(af::span, af::span, b)));
        auto rawTarget = fl::ext::afToVector<int>(target.array()(af::span, b));

        std::cout << "\n|T|: ";
        for (auto& word : rawTarget) {
          std::cout << tokenDictionary_.getEntry(word) << " ";
        }
        std::cout << "\n|P|: ";
        for (auto& word : viterbiPath) {
          std::cout << tokenDictionary_.getEntry(word) << " ";
        }
        std::cout << "\n---";
      }
    }
  }
}

void Trainer::evalASR() {
  FL_LOG_MASTER(INFO) << "ASR evaluation started.";
  asrFrontEnd_->eval();
  lmFrontEnd_->eval();
  encoder_->eval();
  asrCriterionLinear_->eval();
  asrCriterion_->eval();
  lmCriterion_->eval();

  for (const auto& set : asrValidDatasets_) {
    const auto tag = set.first;
    const auto validDataset = set.second;

    int cnt = 0;
    for (const auto& batch : *validDataset) {
      if (cnt++ == 100) {
        break;
      }
      auto input = fl::input(batch[fl::app::asr::kInputIdx]);
      auto target = fl::noGrad(batch[fl::app::asr::kTargetIdx]);
      auto inputSizes = batch[fl::app::asr::kDurationIdx];

      auto output =
          forwardSequentialModuleWithPadMask(input, asrFrontEnd_, inputSizes);
      output = ext::forwardSequentialModuleWithPadMask(
          output, encoder_, inputSizes, false, output.dims(1), output.dims(2));
      output = asrCriterionLinear_->forward({output}).front();

      for (int b = 0; b < output.dims(2); ++b) {
        auto viterbiPath = fl::ext::afToVector<int>(
            asrCriterion_->viterbiPath(output.array()(af::span, af::span, b)));
        auto rawTarget = fl::ext::afToVector<int>(target.array()(af::span, b));

        std::cout << "\n|T|: ";
        for (auto& word : rawTarget) {
          std::cout << tokenDictionary_.getEntry(word) << " ";
        }
        std::cout << "\n|P|: ";
        for (auto& word : viterbiPath) {
          std::cout << tokenDictionary_.getEntry(word) << " ";
        }
        std::cout << "\n---";
      }
    }
  }
}
#endif

} // namespace joint
} // namespace app
} // namespace fl
