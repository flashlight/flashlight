/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/app/asr/augmentation/SoundEffectConfig.h"
#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/common/Flags.h"
#include "flashlight/app/asr/criterion/criterion.h"
#include "flashlight/app/asr/data/FeatureTransforms.h"
#include "flashlight/app/asr/decoder/TranscriptionUtils.h"
#include "flashlight/app/asr/runtime/runtime.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/common/ModulePlugin.h"
#include "flashlight/ext/common/SequentialBuilder.h"
#include "flashlight/ext/common/Serializer.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"

#include "flashlight/app/asr/tutorial/Trainer.h"

using namespace fl::app::asr;

namespace {
DEFINE_string(init_model, "", "[train] Path of a model to fork from");
}

/* ================================ Trainer ================================ */

/* ============= Public functions ============= */
Trainer::Trainer(const std::string& mode) {
  // Parse from Gflags
  if (mode == "train") {
    initTrain();
  } else if (mode == "continue") {
    initContinue();
  } else if (mode == "fork") {
    initFork();
  } else {
    throw std::invalid_argument("Trainer doesn't support mode: " + mode);
  }
  checkArgs();
  gflagsStr_ = serializeGflags();
  FL_LOG_MASTER(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  createDatasets();
  initArrayFire();
  if (FLAGS_enable_distributed) {
    reducer_ = std::make_shared<fl::CoalescingReducer>(
        1.0 / (fl::getWorldSize() * FLAGS_batchsize), true, true);
  }
  experimentDirectory_ = fl::lib::pathsConcat(FLAGS_rundir, FLAGS_runname);
  if (isMaster()) {
    fl::lib::dirCreate(experimentDirectory_);
    logWriter_ = fl::lib::createOutputStream(
        fl::lib::pathsConcat(experimentDirectory_, "log"), std::ios_base::app);
  }

  FL_LOG_MASTER(INFO) << "network (" << fl::numTotalParams(network_)
                      << " params): " << network_->prettyString();
  FL_LOG_MASTER(INFO) << "criterion (" << fl::numTotalParams(criterion_)
                      << " params): " << criterion_->prettyString();
  FL_LOG_MASTER(INFO) << "optimizer: " << networkOptimizer_->prettyString();
}

void Trainer::runTraining() {
  FL_LOG_MASTER(INFO) << "training started (epoch=" << epoch_
                      << " batch=" << batchIdx_ << ")";

  if (reducer_) {
    fl::distributeModuleGrads(network_, reducer_);
    fl::distributeModuleGrads(criterion_, reducer_);
  }
  fl::allReduceParameters(network_);
  fl::allReduceParameters(criterion_);
  auto modelPath = fl::lib::pathsConcat(experimentDirectory_, "model.bin");

  createSpecAugmentation();

  while (batchIdx_ < FLAGS_iter) {
    trainDataset_ = loadPrefetchDataset(
        trainDataset_, FLAGS_nthread, true /* shuffle */, epoch_ /* seed */);

    for (auto& batch : *trainDataset_) {
      // Run train
      runTimeMeter_.resume();
      batchTimerMeter_.resume();
      trainStep(batch);
      batchTimerMeter_.incUnit();
      ++batchIdx_;

      // Run evaluation and save best checkpoint
      if (FLAGS_reportiters && batchIdx_ % FLAGS_reportiters == 0) {
        stopTimers();
        evalStep();
        syncMeters();
        auto progress = getProgress();
        FL_LOG_MASTER(INFO) << progress;
        if (isMaster()) {
          logWriter_ << progress << "\n" << std::flush;
        }
        resetMeters();

        for (const auto& meter : validStatsMeters_) {
          const auto& tag = meter.first;
          auto wer = meter.second.wrdEdit.value()[0];
          if (bestValidWer_.find(tag) == bestValidWer_.end() ||
              wer < bestValidWer_[tag]) {
            bestValidWer_[tag] = wer;
            saveCheckpoint(modelPath, "." + tag);
          }
        }
      }

      // Force saving checkpoint every given interval
      //   if (FLAGS_train_save_updates &&
      //       batchIdx_ % FLAGS_train_save_updates == 0) {
      //     stopTimers();
      //     saveCheckpoint(modelPath, "." + std::to_string(batchIdx_));
      //   }
    }

    // Advance epoch
    stopTimers();
    ++epoch_;
    saveCheckpoint(modelPath);
    logMemoryManagerStatus();
  }
}

void Trainer::trainStep(const std::vector<af::array>& batch) {
  network_->train();
  criterion_->train();
  setLr();

  // 1. Sample
  sampleTimerMeter_.resume();
  auto input = fl::input(batch[kInputIdx]);
  auto target = fl::noGrad(batch[kTargetIdx]);
  auto inputSizes = batch[kDurationIdx];
  auto sampleNames = readSampleIds(batch[kSampleIdx]);
  sampleTimerMeter_.stopAndIncUnit();
  dataStatsMeter_.add(batch[kInputIdx], batch[kTargetIdx]);

  // 2. Forward
  fwdTimeMeter_.resume();
  if (FLAGS_saug_start_update >= 0 && batchIdx_ >= FLAGS_saug_start_update) {
    input = specAug_->forward({input}).front();
  }
  auto output =
      ext::forwardSequentialModuleWithPadMask(input, network_, inputSizes);
  af::sync();
  critFwdTimeMeter_.resume();
  auto loss = criterion_->forward({output, target}).front();
  af::sync();
  fwdTimeMeter_.stopAndIncUnit();
  critFwdTimeMeter_.stopAndIncUnit();
  if (af::anyTrue<bool>(af::isNaN(loss.array())) ||
      af::anyTrue<bool>(af::isInf(loss.array()))) {
    LOG(FATAL) << "Loss has NaN/Inf values. Samples - "
               << lib::join(",", sampleNames);
  }
  trainStatsMeter_.loss.add(af::mean<double>(loss.array()));
  if (hasher_(lib::join(",", sampleNames)) % 100 <= FLAGS_pcttraineval) {
    evalOutput(output.array(), target.array(), trainStatsMeter_);
  }

  // 3. Backward
  bwdTimeMeter_.resume();
  networkOptimizer_->zeroGrad();
  loss.backward();
  if (reducer_) {
    reducer_->finalize();
  }
  af::sync();
  bwdTimeMeter_.stopAndIncUnit();

  // 4. Optimization
  optimTimeMeter_.resume();
  if (FLAGS_maxgradnorm > 0) {
    fl::clipGradNorm(network_->params(), FLAGS_maxgradnorm);
  }
  networkOptimizer_->step();
  af::sync();
  optimTimeMeter_.stopAndIncUnit();
}

void Trainer::evalStep() {
  network_->eval();
  criterion_->eval();

  for (const auto& set : validDatasets_) {
    const auto tag = set.first;
    const auto validDataset = set.second;
    auto& validMeter = validStatsMeters_[tag];
    validMeter.tknEdit.reset();
    validMeter.wrdEdit.reset();
    validMeter.loss.reset();

    for (const auto& batch : *validDataset) {
      auto input = fl::input(batch[kInputIdx]);
      auto target = fl::noGrad(batch[kTargetIdx]);
      auto inputSizes = batch[kDurationIdx];

      auto output =
          ext::forwardSequentialModuleWithPadMask(input, network_, inputSizes);
      auto loss = criterion_->forward({output, target}).front();

      validMeter.loss.add(af::mean<double>(loss.array()));
      evalOutput(output.array(), target.array(), validMeter);
    }
  }
}

/* ============= Initializers ============= */
void Trainer::initTrain() {
  FL_LOG_MASTER(INFO) << "Creating a fresh model";

  createDictionary();
  createNetwork();
  createCriterion();
  createOptimizer();
}

void Trainer::initContinue() {
  auto checkPoint = fl::lib::pathsConcat(experimentDirectory_, "model.bin");
  if (!fl::lib::fileExists(checkPoint)) {
    throw std::invalid_argument(
        "Checkpoint doesn't exist to continue training: " + checkPoint);
  }
  FL_LOG_MASTER(INFO) << "Continue training from file: " << checkPoint;
  fl::ext::Serializer::load(
      checkPoint,
      version_,
      network_,
      criterion_,
      bestValidWer_,
      networkOptimizer_,
      epoch_,
      batchIdx_,
      gflagsStr_);

  // overwrite flags using the ones from command line
  gflags::ReadFlagsFromString(gflagsStr_, gflags::GetArgv0(), true);

  createDictionary();
  // the network, criterion and optimizer will be reused
}

void Trainer::initFork() {
  if (!fl::lib::fileExists(FLAGS_init_model)) {
    throw std::invalid_argument(
        "Checkpoint doesn't exist for finetuning: " + FLAGS_init_model);
  }
  FL_LOG_MASTER(INFO) << "Fork training from file: " << FLAGS_init_model;

  std::shared_ptr<fl::FirstOrderOptimizer> dummyOptimizer;
  fl::ext::Serializer::load(
      FLAGS_init_model,
      version_,
      network_,
      criterion_,
      dummyOptimizer,
      epoch_,
      batchIdx_);

  createDictionary();
  createOptimizer();
  // the network and criterion will be reused
}

void Trainer::createDictionary() {
  auto dictPath = fl::lib::pathsConcat(FLAGS_tokensdir, FLAGS_tokens);
  if (dictPath.empty() || !fl::lib::fileExists(dictPath)) {
    throw std::runtime_error(
        "Invalid dictionary filepath specified with "
        "--tokensdir and --tokens: \"" +
        dictPath + "\"");
  }
  tokenDictionary_ = fl::lib::text::Dictionary(dictPath);
  tokenDictionary_.addEntry(kBlankToken);
  numClasses_ = tokenDictionary_.indexSize();
  FL_LOG_MASTER(INFO) << "Number of tokens: " << numClasses_;

  if (!FLAGS_lexicon.empty()) {
    lexicon_ = fl::lib::text::loadWords(FLAGS_lexicon, FLAGS_maxword);
    wordDictionary_ = fl::lib::text::createWordDict(lexicon_);
    FL_LOG_MASTER(INFO) << "Number of words: " << wordDictionary_.indexSize();
  }
}

void Trainer::createDatasets() {
  fl::lib::audio::FeatureParams featParams(
      FLAGS_samplerate,
      FLAGS_framesizems,
      FLAGS_framestridems,
      FLAGS_filterbanks,
      FLAGS_lowfreqfilterbank,
      FLAGS_highfreqfilterbank,
      FLAGS_mfcccoeffs,
      kLifterParam /* lifterparam */,
      FLAGS_devwin /* delta window */,
      FLAGS_devwin /* delta-delta window */);
  featParams.useEnergy = false;
  featParams.usePower = false;
  featParams.zeroMeanFrame = false;
  numFeatures_ = -1;
  FeatureType featType = FeatureType::NONE;
  if (FLAGS_pow) {
    featType = FeatureType::POW_SPECTRUM;
    numFeatures_ = featParams.powSpecFeatSz();
  } else if (FLAGS_mfsc) {
    featType = FeatureType::MFSC;
    numFeatures_ = featParams.mfscFeatSz();
  } else if (FLAGS_mfcc) {
    featType = FeatureType::MFCC;
    numFeatures_ = featParams.mfccFeatSz();
  }
  TargetGenerationConfig targetGenConfig(
      FLAGS_wordseparator,
      FLAGS_sampletarget,
      FLAGS_criterion,
      FLAGS_surround,
      FLAGS_eostoken,
      FLAGS_replabel,
      true /* skip unk */,
      FLAGS_usewordpiece /* fallback2LetterWordSepLeft */,
      !FLAGS_usewordpiece /* fallback2LetterWordSepLeft */);

  const auto sfxConf = (FLAGS_sfx_config.empty())
      ? std::vector<sfx::SoundEffectConfig>()
      : sfx::readSoundEffectConfigFile(FLAGS_sfx_config);

  auto inputTransform = inputFeatures(
      featParams,
      featType,
      {FLAGS_localnrmlleftctx, FLAGS_localnrmlrightctx},
      sfxConf);
  auto targetTransform =
      targetFeatures(tokenDictionary_, lexicon_, targetGenConfig);
  auto wordTransform = wordFeatures(wordDictionary_);
  int targetpadVal = FLAGS_eostoken
      ? tokenDictionary_.getIndex(fl::app::asr::kEosToken)
      : kTargetPadValue;
  int wordpadVal = kTargetPadValue;

  std::vector<std::string> trainSplits = fl::lib::split(",", FLAGS_train, true);
  trainDataset_ = createDataset(
      trainSplits,
      FLAGS_datadir,
      FLAGS_batchsize,
      inputTransform,
      targetTransform,
      wordTransform,
      std::make_tuple(0, targetpadVal, wordpadVal),
      fl::getWorldRank(),
      fl::getWorldSize());

  int64_t validBatchSize =
      FLAGS_validbatchsize == -1 ? FLAGS_batchsize : FLAGS_validbatchsize;
  auto validSets = fl::lib::split(',', fl::lib::trim(FLAGS_valid), true);
  for (const auto& s : validSets) {
    // assume the format is tag:filepath
    auto parts = fl::lib::splitOnAnyOf(":", s);
    std::string tag, path;
    if (parts.size() == 1) {
      tag = parts[0];
      path = parts[0];
    } else if (parts.size() == 2) {
      tag = parts[0];
      path = parts[0];
    } else {
      LOG(FATAL) << "invalid valid set: " << s;
    }

    validDatasets_[tag] = createDataset(
        {path},
        FLAGS_datadir,
        validBatchSize,
        inputTransform,
        targetTransform,
        wordTransform,
        std::make_tuple(0, targetpadVal, wordpadVal),
        fl::getWorldRank(),
        fl::getWorldSize());
    validStatsMeters_[tag] = DatasetMeters();
  }
}

void Trainer::createNetwork() {
  auto archfile = fl::lib::pathsConcat(FLAGS_archdir, FLAGS_arch);
  FL_LOG_MASTER(INFO) << "Loading architecture file from " << archfile;
  // Encoder network, works on audio
  if (fl::lib::endsWith(archfile, ".so")) {
    network_ = fl::ext::ModulePlugin(archfile).arch(numFeatures_, numClasses_);
  } else {
    network_ =
        fl::ext::buildSequentialModule(archfile, numFeatures_, numClasses_);
  }
  FL_LOG_MASTER(INFO) << "[Network] " << network_->prettyString();
  FL_LOG_MASTER(INFO) << "[Network Params: " << numTotalParams(network_) << "]";
}

void Trainer::createCriterion() {
  auto scalemode = getCriterionScaleMode(FLAGS_onorm, FLAGS_sqnorm);
  criterion_ = std::make_shared<CTCLoss>(scalemode);
  FL_LOG_MASTER(INFO) << "[Criterion] " << criterion_->prettyString();
}

void Trainer::createOptimizer() {
  networkOptimizer_ = initOptimizer(
      {network_}, FLAGS_netoptim, FLAGS_lr, FLAGS_momentum, FLAGS_weightdecay);
  FL_LOG_MASTER(INFO) << "[Network Optimizer] "
                      << networkOptimizer_->prettyString();
}

void Trainer::createSpecAugmentation() {
  if (FLAGS_saug_start_update >= 0) {
    if (!(FLAGS_pow || FLAGS_mfsc || FLAGS_mfcc)) {
      specAug_ = std::make_shared<fl::RawWavSpecAugment>(
          FLAGS_filterbanks,
          FLAGS_saug_fmaskf,
          FLAGS_saug_fmaskn,
          FLAGS_saug_tmaskt,
          FLAGS_saug_tmaskp,
          FLAGS_saug_tmaskn,
          FLAGS_filterbanks,
          FLAGS_lowfreqfilterbank,
          FLAGS_highfreqfilterbank,
          FLAGS_samplerate);
    } else {
      specAug_ = std::make_shared<fl::SpecAugment>(
          FLAGS_filterbanks,
          FLAGS_saug_fmaskf,
          FLAGS_saug_fmaskn,
          FLAGS_saug_tmaskt,
          FLAGS_saug_tmaskp,
          FLAGS_saug_tmaskn);
    }
  }
}

/* ============= Stateful training helpers ============= */
void Trainer::setLr() {
  if (batchIdx_ < FLAGS_warmup) {
    // warmup stage
    lr_ = FLAGS_lr * batchIdx_ / (double(FLAGS_warmup));
  } else {
    int64_t epochsAfterDecay = epoch_ - FLAGS_lr_decay;
    double lrDecayScale = std::pow(
        0.5,
        (epochsAfterDecay < 0 ? 0
                              : 1 + epochsAfterDecay / FLAGS_lr_decay_step));

    lr_ = FLAGS_lr * lrDecayScale;
  }
  networkOptimizer_->setLr(lr_);
}

void Trainer::evalOutput(
    const af::array& output,
    const af::array& target,
    DatasetMeters& meter) {
  auto batchSize = output.dims(2);
  for (int b = 0; b < batchSize; ++b) {
    auto viterbiPath = fl::ext::afToVector<int>(
        criterion_->viterbiPath(output(af::span, af::span, b)));
    auto rawTarget = fl::ext::afToVector<int>(target(af::span, b));

    // Remove `-1`s appended to the target for batching (if any)
    auto labellen = getTargetSize(rawTarget.data(), rawTarget.size());
    rawTarget.resize(labellen);

    // remap actual, predicted targets for evaluating edit distance error
    auto letterPrediction = tknPrediction2Ltr(
        viterbiPath,
        tokenDictionary_,
        FLAGS_criterion,
        FLAGS_surround,
        FLAGS_eostoken,
        FLAGS_replabel,
        FLAGS_usewordpiece,
        FLAGS_wordseparator);
    auto letterTarget = tknTarget2Ltr(
        rawTarget,
        tokenDictionary_,
        FLAGS_criterion,
        FLAGS_surround,
        FLAGS_eostoken,
        FLAGS_replabel,
        FLAGS_usewordpiece,
        FLAGS_wordseparator);

    auto wordPrediction = tkn2Wrd(letterPrediction, FLAGS_wordseparator);
    auto wordTarget = tkn2Wrd(letterTarget, FLAGS_wordseparator);

    meter.tknEdit.add(letterPrediction, letterTarget);
    meter.wrdEdit.add(wordPrediction, wordTarget);
  }
}

/* ============= Stateless training helpers ============= */
void Trainer::initArrayFire() const {
  // Set arrayfire seed for reproducibility
  af::setSeed(FLAGS_seed);
}

bool Trainer::isMaster() const {
  return fl::getWorldRank() == 0;
}

void Trainer::checkArgs() const {
  if (version_ != FL_APP_ASR_VERSION) {
    FL_LOG_MASTER(INFO) << "Model version (" << version_
                        << ") does not match FL_APP_LM_VERSION ("
                        << FL_APP_ASR_VERSION << ")";
  }
}

/* ============= Meter helpers ============= */
void Trainer::resetMeters() {
  trainStatsMeter_.tknEdit.reset();
  trainStatsMeter_.wrdEdit.reset();
  trainStatsMeter_.loss.reset();

  runTimeMeter_.reset();
  batchTimerMeter_.reset();
  sampleTimerMeter_.reset();
  fwdTimeMeter_.reset();
  critFwdTimeMeter_.reset();
  bwdTimeMeter_.reset();
  optimTimeMeter_.reset();
  dataStatsMeter_.reset();
}

void Trainer::syncMeters() {
  fl::ext::syncMeter(trainStatsMeter_.tknEdit);
  fl::ext::syncMeter(trainStatsMeter_.wrdEdit);
  fl::ext::syncMeter(trainStatsMeter_.loss);
  for (auto& meter : validStatsMeters_) {
    auto& validMeter = meter.second;
    fl::ext::syncMeter(validMeter.tknEdit);
    fl::ext::syncMeter(validMeter.wrdEdit);
    fl::ext::syncMeter(validMeter.loss);
  }
  fl::ext::syncMeter(runTimeMeter_);
  fl::ext::syncMeter(batchTimerMeter_);
  fl::ext::syncMeter(sampleTimerMeter_);
  fl::ext::syncMeter(fwdTimeMeter_);
  fl::ext::syncMeter(critFwdTimeMeter_);
  fl::ext::syncMeter(bwdTimeMeter_);
  fl::ext::syncMeter(optimTimeMeter_);
  fl::ext::syncMeter(dataStatsMeter_);
}

void Trainer::stopTimers() {
  runTimeMeter_.stop();
  batchTimerMeter_.stop();
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

  FL_LOG_MASTER(INFO) << "saving model checkpoint (epoch=" << epoch_
                      << " batch=" << batchIdx_ << ") to: " << path;
  fl::ext::Serializer::save(
      path,
      FL_APP_ASR_VERSION,
      network_,
      criterion_,
      bestValidWer_,
      networkOptimizer_,
      epoch_,
      batchIdx_,
      gflagsStr_);

  if (!suffix.empty()) {
    fl::ext::Serializer::save(
        path + suffix,
        FL_APP_ASR_VERSION,
        network_,
        criterion_,
        bestValidWer_,
        networkOptimizer_,
        epoch_,
        batchIdx_,
        gflagsStr_);
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
  insertItem("timestamp", lib::getCurrentDate() + " " + lib::getCurrentTime());
  insertItem("epoch", format("%8d", epoch_));
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
  insertItem("bch(ms)", format("%.2f", batchTimerMeter_.value() * 1000));
  insertItem("smp(ms)", format("%.2f", sampleTimerMeter_.value() * 1000));
  insertItem("fwd(ms)", format("%.2f", fwdTimeMeter_.value() * 1000));
  insertItem("crit-fwd(ms)", format("%.2f", critFwdTimeMeter_.value() * 1000));
  insertItem("bwd(ms)", format("%.2f", bwdTimeMeter_.value() * 1000));
  insertItem("optim(ms)", format("%.2f", optimTimeMeter_.value() * 1000));

  insertItem("loss", format("%10.5f", trainStatsMeter_.loss.value()[0]));
  insertItem(
      "train-TER", format("%5.2f", trainStatsMeter_.tknEdit.errorRate()[0]));
  insertItem(
      "train-WER", format("%5.2f", trainStatsMeter_.wrdEdit.errorRate()[0]));

  for (const auto& meter : validStatsMeters_) {
    const auto tag = meter.first;
    const auto& validMeter = meter.second;
    insertItem(tag + "-loss", format("%10.5f", validMeter.loss.value()[0]));
    insertItem(
        tag + "-TER", format("%5.2f", validMeter.tknEdit.errorRate()[0]));
    insertItem(
        tag + "-WER", format("%5.2f", validMeter.wrdEdit.errorRate()[0]));
  }

  auto stats = dataStatsMeter_.value();
  auto numsamples = std::max<int64_t>(stats[4], 1);
  auto isztotal = stats[0];
  auto tsztotal = stats[1];
  auto tszmax = stats[3];
  insertItem("avg-isz", format("%03d", isztotal / numsamples));
  insertItem("avg-tsz", format("%03d", tsztotal / numsamples));
  insertItem("max-tsz", format("%03d", tszmax));

  double audioProcSec = isztotal * FLAGS_batchsize;
  if (FLAGS_pow || FLAGS_mfcc || FLAGS_mfsc) {
    audioProcSec = audioProcSec * FLAGS_framestridems / 1000.0;
  } else {
    audioProcSec /= FLAGS_samplerate;
  }
  auto worldSize = fl::getWorldSize();
  double timeTakenSec = batchTimerMeter_.value() * numsamples / worldSize;

  insertItem("hrs", format("%7.2f", audioProcSec / 3600.0));
  insertItem(
      "thrpt(sec/sec)",
      timeTakenSec > 0.0 ? format("%.2f", audioProcSec / timeTakenSec) : "n/a");
  return status;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage(
      "Usage: \n " + exec + " train [flags]\n or " + exec +
      " continue [directory] [flags]\n or " + exec +
      " fork [directory/model] [flags]");
  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  /* Parse or load persistent states */
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
  }
  if (FLAGS_enable_distributed) {
    fl::ext::initDistributed(
        FLAGS_world_rank,
        FLAGS_world_size,
        FLAGS_max_devices_per_node,
        FLAGS_rndv_filepath);
  }

  /* Run train */
  Trainer trainer(argv[1]);
  // flags may be overridden from the model
  // so reloading from command line again
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  trainer.runTraining();
}
