/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/lm/Trainer.h"
#include <algorithm>

using namespace fl::ext;
using namespace fl::lib;

namespace fl {
namespace app {
namespace lm {

/* ================================ FLAGS ================================ */

/* CRITERION OPTIONS */
DEFINE_string(
    loss_type,
    "adsm",
    "Loss type during optimization. \
    Supported for now: adaptive softmax (adsm) and cross entropy (ce).");
DEFINE_int64(
    loss_adsm_input_size,
    0,
    "Input size of AdaptiveSoftMax (i.e. output size of network).");
DEFINE_string(
    loss_adsm_cutoffs,
    "",
    "Cutoffs for AdaptiveSoftMax comma separated.");

/* DISTRIBUTED TRAINING */
DEFINE_bool(distributed_enable, false, "Enable distributed training.");
DEFINE_int64(
    distributed_world_rank,
    0,
    "Distributed training. Rank of the process (Used if rndv_filepath is not empty).");
DEFINE_int64(
    distributed_world_size,
    1,
    "Distributed training. Total number of the process (Used if rndv_filepath is not empty).");
DEFINE_int64(
    distributed_max_devices_per_node,
    8,
    "Distributed training. The maximum number of devices per training node.");
DEFINE_string(
    distributed_rndv_filepath,
    "",
    "Distributed training. Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");

/* RUN OPTIONS */
DEFINE_string(
    exp_rundir,
    "",
    "Experiment path, where logs and models will be stored.");
DEFINE_string(
    exp_model_name,
    "model",
    "Name used to save model bin and model log. '--exp_rundir' will be used as prefix.");
DEFINE_string(
    exp_init_model_path,
    "",
    "Initialization model full path, used as init model to start training.");

/* DATA OPTIONS */
DEFINE_string(
    data_dir,
    "",
    "Prefix for the 'data_train' and 'data_valid' files.");
DEFINE_string(
    data_train,
    "",
    "Comma-separated list of training data files; '--data_dir' will be used to add prefix for the files.");
DEFINE_string(
    data_valid,
    "",
    "Comma-separated list of validation/test data files; \
    '--data_dir' will be used to add prefix for the files.");
DEFINE_int64(
    data_batch_size,
    1,
    "Batch size of data (per process in distributed training). \
    If '--data_use_dynamic_batching=true' is used can be different \
    to have '--data_tokens_per_sample' * '--data_batch_size' tokens in the batch.");
DEFINE_int64(
    data_tokens_per_sample,
    1024,
    "Max number of tokens per sample in the data. \
    See details for '--data_sample_break_mode' and '--data_use_dynamic_batching'.");
DEFINE_string(
    data_sample_break_mode,
    "none",
    "How to split sentences to form samples and batch. \
    'none' means split joined text into chunks of '--data_tokens_per_sample' tokens. \
    'eos' means split text by the end of sentence to create a sample, \
    if sentence len greater than '--data_tokens_per_sample' then exclude this sentence.");
DEFINE_bool(
    data_use_dynamic_batching,
    false,
    "if or not use dynamic batching in case of '--data_sample_break_mode=eos'.");

/* DICTIONARY OPTIONS */
DEFINE_string(
    dictionary,
    "",
    "Path to the dictionary file (read/write), which defines tokens set of language model.");
DEFINE_int64(
    dictionary_max_size,
    -1,
    "Number of rows to use from the dictionary file (top rows), cutting the number of target classes.");

/* TRAIN OPTIONS */
DEFINE_string(train_task, "autoreg", "Task for training: autoreg or mask");
DEFINE_string(
    train_arch_file,
    "model.so",
    "Network .cpp architecture file path");
DEFINE_int64(
    train_seed,
    0,
    "Manually specify Arrayfire seed for reproducibility.");
DEFINE_string(
    train_optimizer,
    "nag",
    "Optimizer to use in training. Supported for now: adagrad, sgd, nag.");

DEFINE_int64(
    train_warmup_updates,
    0,
    "Use warmup. Ramp learning rate from '--train_warmup_init_lr' till '--train_lr' \
    linearly during '--train_warmup_updates' number of updates.");
DEFINE_double(
    train_warmup_init_lr,
    0.0,
    "Warmup init learning rate from which ramping starts.");

DEFINE_double(
    train_lr,
    1.0,
    "Learning rate for optimization process. \
    If '--train_warmup' is used then lr is warmuped to '--train_lr' value.");
DEFINE_string(
    train_lr_schedule,
    "fixed",
    "Learning rate schedule. \
    'fixed' means 'train_lr' will be used all the time (except warmup stage); \
    'invsqrt' means decaying with respect to q/sqrt(nUpdates).");
DEFINE_double(
    train_momentum,
    0.0,
    "Momentum factor used in optimization process.");
DEFINE_double(
    train_weight_decay,
    0.0,
    "L2 penalty coefficient for the parameters during optimization process.");

DEFINE_double(
    train_max_grad_norm,
    0.0,
    "Clip gradients at this value (0 = no clipping).");
DEFINE_int64(
    train_save_updates,
    0,
    "Specifies to save model every '--train_save_updates' updates.");
DEFINE_int64(
    train_report_updates,
    0,
    "Number of updates after which we will run evaluation and save model, \
    if 0 we only do this at the end of epoch ");
DEFINE_int64(
    train_total_updates,
    std::numeric_limits<int64_t>::max(),
    "Total number of updates.");

/* MASK OPTIONS */
DEFINE_double(mask_prob, 0.15, "[mask lm task] Probability of masking.");
DEFINE_double(
    mask_rand_token_prob,
    0.1,
    "[mask lm task] Probability of mask token to set random token.");
DEFINE_double(
    mask_same_token_prob,
    0.1,
    "[mask lm task] Probability of mask token to set original token.");
DEFINE_int64(
    mask_min_length,
    1,
    "[mask lm task] Min number of masked tokens in each sample.");

/* ================================ Trainer ================================ */

/* ============= Public functions ============= */
Trainer::Trainer(const std::string& mode) {
  // Parse from Gflags
  (void)fl::ext::ModulePlugin(FLAGS_train_arch_file);
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
  gflagsStr_ = fl::ext::serializeGflags();
  FL_LOG_MASTER(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  initArrayFire();
  if (FLAGS_distributed_enable) {
    reducer_ = std::make_shared<fl::CoalescingReducer>(1.0, true, true);
  }

  FL_LOG_MASTER(INFO) << "network (" << fl::numTotalParams(network_)
                      << " params): " << network_->prettyString();
  FL_LOG_MASTER(INFO) << "criterion (" << fl::numTotalParams(criterion_)
                      << " params): " << criterion_->prettyString();
  if (optimizer_) {
    FL_LOG_MASTER(INFO) << "optimizer: " << optimizer_->prettyString();
  }
}

void Trainer::runTraining() {
  if (isMaster()) {
    dirCreate(FLAGS_exp_rundir);
    logWriter_ = createOutputStream(
        pathsConcat(FLAGS_exp_rundir, FLAGS_exp_model_name + ".log"),
        std::ios_base::app);
  }

  FL_LOG_MASTER(INFO) << "training started (epoch=" << epoch_
                      << " batch=" << batchIdx_ << ")";

  fl::allReduceParameters(network_);
  fl::allReduceParameters(criterion_);
  auto modelPath = pathsConcat(FLAGS_exp_rundir, FLAGS_exp_model_name + ".bin");

  while (batchIdx_ < FLAGS_train_total_updates) {
    // Advance epoch
    if (batchIdx_ && batchIdx_ % trainDataset_->size() == 0) {
      stopTimers();
      ++epoch_;
      trainDataset_->shuffle(FLAGS_train_seed + epoch_);
      saveCheckpoint(modelPath);
      logMemoryManagerStatus();
    }

    // Run train
    runTimeMeter_.resume();
    batchTimerMeter_.resume();
    trainStep();
    batchTimerMeter_.incUnit();
    ++batchIdx_;

    // Run evaluation and save best checkpoint
    if (FLAGS_train_report_updates &&
        batchIdx_ % FLAGS_train_report_updates == 0) {
      auto loss = runEvaluation();
      if (loss < bestLoss_) {
        bestLoss_ = loss;
        saveCheckpoint(modelPath, ".best");
      }

      auto progress = getProgress();
      FL_LOG_MASTER(INFO) << progress;
      if (isMaster()) {
        logWriter_ << progress << "\n" << std::flush;
      }
      resetMeters();
    }

    // Force saving checkpoint every given interval
    if (FLAGS_train_save_updates && batchIdx_ % FLAGS_train_save_updates == 0) {
      stopTimers();
      saveCheckpoint(modelPath, "." + std::to_string(batchIdx_));
    }
  }
}

void Trainer::trainStep() {
  network_->train();
  criterion_->train();
  setLr();

  // 1. Sample
  fl::Variable input, target;
  sampleTimerMeter_.resume();
  std::tie(input, target) = getInputAndTarget(trainDataset_->get(batchIdx_));
  af::array inputSizes = af::sum(input.array() != kPadIdx_, 0);
  sampleTimerMeter_.stopAndIncUnit();

  // 2. Forward
  fwdTimeMeter_.resume();
  auto output = network_->forward({input, fl::noGrad(inputSizes)}).front();
  af::sync();
  critFwdTimeMeter_.resume();
  auto loss = criterion_->forward({output, target}).front();
  af::sync();
  fwdTimeMeter_.stopAndIncUnit();
  critFwdTimeMeter_.stopAndIncUnit();

  float numTokens = af::count<float>(target.array() != kPadIdx_);
  if (numTokens > 0) {
    auto weight =
        numTokens / (FLAGS_data_tokens_per_sample * FLAGS_data_batch_size);
    trainLossMeter_.add(af::mean<float>(loss.array()) / numTokens, weight);
    tokenCountMeter_.add(numTokens);
  }

  // 3. Backward
  bwdTimeMeter_.resume();
  optimizer_->zeroGrad();
  af::array numTokensArr = af::array(1, &numTokens);
  if (FLAGS_distributed_enable) {
    fl::allReduce(numTokensArr);
  }
  loss = loss / fl::Variable(numTokensArr, false);
  loss.backward();
  reduceGrads();
  af::sync();
  bwdTimeMeter_.stopAndIncUnit();

  // 4. Optimization
  optimTimeMeter_.resume();
  fl::clipGradNorm(parameters_, FLAGS_train_max_grad_norm);
  optimizer_->step();
  af::sync();
  optimTimeMeter_.stopAndIncUnit();
}

void Trainer::evalStep() {
  network_->eval();
  criterion_->eval();

  for (const auto& sample : *validDataset_) {
    fl::Variable input, target;
    std::tie(input, target) = getInputAndTarget(sample);
    af::array inputSizes = af::sum(input.array() != kPadIdx_, 0);
    auto output = network_->forward({input, fl::noGrad(inputSizes)}).front();
    auto loss = criterion_->forward({output, target}).front();
    auto numTokens = af::count<int>(target.array() != kPadIdx_);
    if (numTokens > 0) {
      auto weight = numTokens /
          static_cast<double>(
                        FLAGS_data_tokens_per_sample * FLAGS_data_batch_size);
      validLossMeter_.add(af::mean<double>(loss.array()) / numTokens, weight);
    }
  }
}

float Trainer::runEvaluation() {
  stopTimers();
  evalStep();
  syncMeters();
  auto loss = validLossMeter_.value()[0];
  return loss;
}

/* ============= Initializers ============= */
void Trainer::initTrain() {
  FL_LOG_MASTER(INFO) << "Creating a fresh model";
  createDictionary();
  createNetwork();
  createCriterion();
  createOptimizer();

  createTrainDatasets();
  createValidDatasets();
}

void Trainer::initContinue() {
  auto checkPoint =
      pathsConcat(FLAGS_exp_rundir, FLAGS_exp_model_name + ".bin");
  if (!fileExists(checkPoint)) {
    throw std::invalid_argument(
        "Checkpoint doesn't exist to continue training: " + checkPoint);
  }
  FL_LOG_MASTER(INFO) << "Continue training from file: " << checkPoint;
  fl::ext::Serializer::load(
      checkPoint,
      version_,
      network_,
      criterion_,
      optimizer_,
      epoch_,
      batchIdx_,
      gflagsStr_);

  // overwrite flags using the ones from command line
  gflags::ReadFlagsFromString(gflagsStr_, gflags::GetArgv0(), true);

  createDictionary();
  createTrainDatasets();
  createValidDatasets();
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
      network_,
      criterion_,
      dummyOptimizer,
      epoch_,
      batchIdx_);

  createDictionary();
  createOptimizer();
  createTrainDatasets();
  createValidDatasets();
  // the network and criterion will be reused
}

void Trainer::initEval() {
  if (!fileExists(FLAGS_exp_init_model_path)) {
    throw std::invalid_argument(
        "Checkpoint doesn't exist for evaluation: " +
        FLAGS_exp_init_model_path);
  }
  FL_LOG_MASTER(INFO) << "Evaluate from file: " << FLAGS_exp_init_model_path;

  fl::ext::Serializer::load(
      FLAGS_exp_init_model_path, version_, network_, criterion_);

  createDictionary();
  createValidDatasets();
  // the network and criterion will be reused
}

void Trainer::createDictionary() {
  auto stream = createInputStream(FLAGS_dictionary);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty()) {
      continue;
    }
    auto tkns = splitOnWhitespace(line, true);
    if (tkns.empty()) {
      continue;
    }
    dictionary_.addEntry(tkns.front());
    if (dictionary_.entrySize() == FLAGS_dictionary_max_size &&
        FLAGS_dictionary_max_size > 0) {
      break;
    }
  }
  if (!dictionary_.isContiguous()) {
    throw std::runtime_error("Invalid dictionary_ format - not contiguous");
  }
  kPadIdx_ = dictionary_.getIndex(fl::lib::text::kPadToken);
  kEosIdx_ = dictionary_.getIndex(fl::lib::text::kEosToken);
  kUnkIdx_ = dictionary_.getIndex(fl::lib::text::kUnkToken);
  kMaskIdx_ = dictionary_.getIndex(fl::lib::text::kMaskToken);
  dictionary_.setDefaultIndex(dictionary_.getIndex(fl::lib::text::kUnkToken));
}

void Trainer::createTrainDatasets() {
  fl::lib::text::Tokenizer tokenizer;
  fl::lib::text::PartialFileReader partialFileReader(
      fl::getWorldRank(), fl::getWorldSize());
  trainDataset_ = std::make_shared<TextDataset>(
      FLAGS_data_dir,
      FLAGS_data_train,
      partialFileReader,
      tokenizer,
      dictionary_,
      FLAGS_data_tokens_per_sample,
      FLAGS_data_batch_size,
      FLAGS_data_sample_break_mode,
      true);
  FL_LOG_MASTER(INFO) << "train dataset: " << trainDataset_->size()
                      << " samples";
}

void Trainer::createValidDatasets() {
  fl::lib::text::Tokenizer tokenizer;
  fl::lib::text::PartialFileReader partialFileReader(
      fl::getWorldRank(), fl::getWorldSize());

  validDataset_ = std::make_shared<TextDataset>(
      FLAGS_data_dir,
      FLAGS_data_valid,
      partialFileReader,
      tokenizer,
      dictionary_,
      FLAGS_data_tokens_per_sample,
      FLAGS_data_batch_size,
      "eos",
      FLAGS_data_use_dynamic_batching);
  FL_LOG_MASTER(INFO) << "valid dataset: " << validDataset_->size()
                      << " samples";
}

void Trainer::createNetwork() {
  if (dictionary_.entrySize() == 0) {
    throw std::runtime_error("Dictionary is empty, number of classes is zero");
  }
  network_ = fl::ext::ModulePlugin(FLAGS_train_arch_file)
                 .arch(0, dictionary_.entrySize());
}

void Trainer::createCriterion() {
  if (FLAGS_loss_type == "adsm") {
    if (dictionary_.entrySize() == 0) {
      throw std::runtime_error(
          "Dictionary is empty, number of classes is zero");
    }
    auto softmax = std::make_shared<fl::AdaptiveSoftMax>(
        FLAGS_loss_adsm_input_size, parseCutoffs(dictionary_.entrySize()));
    criterion_ = std::make_shared<fl::AdaptiveSoftMaxLoss>(
        softmax, fl::ReduceMode::SUM, kPadIdx_);
  } else if (FLAGS_loss_type == "ce") {
    criterion_ = std::make_shared<fl::CategoricalCrossEntropy>(
        fl::ReduceMode::SUM, kPadIdx_);
  } else {
    throw std::runtime_error(
        "Criterion is not supported, check 'loss_type' flag possible values");
  }
}

void Trainer::collectParameters() {
  parameters_ = network_->params();
  const auto& criterionParams = criterion_->params();
  parameters_.insert(
      parameters_.end(), criterionParams.begin(), criterionParams.end());
}

void Trainer::createOptimizer() {
  collectParameters();
  if (FLAGS_train_optimizer == "nag") {
    optimizer_ = std::make_shared<fl::NAGOptimizer>(
        parameters_,
        FLAGS_train_lr,
        FLAGS_train_momentum,
        FLAGS_train_weight_decay);
  } else if (FLAGS_train_optimizer == "sgd") {
    optimizer_ = std::make_shared<fl::SGDOptimizer>(
        parameters_,
        FLAGS_train_lr,
        FLAGS_train_momentum,
        FLAGS_train_weight_decay,
        false);
  } else if (FLAGS_train_optimizer == "adagrad") {
    optimizer_ = std::make_shared<fl::AdagradOptimizer>(
        parameters_, FLAGS_train_lr, 1e-8, FLAGS_train_weight_decay);
  } else {
    throw std::runtime_error(
        "Optimizer is not supported, check 'train_optimizer' flag possible values");
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
    // set min length of the masked tokens
    int nTotalMask =
        std::max(int(FLAGS_mask_prob * T), (int)FLAGS_mask_min_length);
    // set total mask
    af::array totalMask = randMatrixSortedIndices < nTotalMask;
    af::array notMasked = !totalMask;
    af::array woMaskTokenMask = randMatrixSortedIndices <
        (FLAGS_mask_rand_token_prob + FLAGS_mask_same_token_prob) * nTotalMask;
    af::array randMask =
        randMatrixSortedIndices < FLAGS_mask_rand_token_prob * nTotalMask;

    inputMasked(totalMask) = kMaskIdx_;
    inputMasked(woMaskTokenMask) = af::flat(sample[0])(woMaskTokenMask);
    if (af::anyTrue<bool>(randMask)) {
      // exclude 4 special tokens from the consideration: pad, eos, unk and
      // mask
      std::vector<int> specialTokens = {
          kPadIdx_, kEosIdx_, kUnkIdx_, kMaskIdx_};
      std::sort(specialTokens.begin(), specialTokens.end());
      auto randVals = (af::randu(af::sum(randMask).scalar<unsigned int>()) *
                       (dictionary_.entrySize() - 1 - specialTokens.size()))
                          .as(s32);
      for (auto specialVal : specialTokens) {
        auto specialMask = randVals >= specialVal;
        randVals(specialMask) = randVals(specialMask) + 1;
      }
      inputMasked(randMask) = randVals;
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
  double lr;
  if (batchIdx_ < FLAGS_train_warmup_updates) {
    // warmup stage
    lr = FLAGS_train_warmup_init_lr +
        (FLAGS_train_lr - FLAGS_train_warmup_init_lr) * batchIdx_ /
            (double(FLAGS_train_warmup_updates));
  } else {
    if (FLAGS_train_lr_schedule == "fixed") {
      // after warmup stage + fixed policy
      lr = FLAGS_train_lr;
    } else if (FLAGS_train_lr_schedule == "invsqrt") {
      // after warmup stage + invsqrt policy
      if (FLAGS_train_warmup_updates > 0) {
        lr = FLAGS_train_lr * std::sqrt(FLAGS_train_warmup_updates) /
            std::sqrt(batchIdx_);
      } else {
        lr = FLAGS_train_lr / std::sqrt(batchIdx_ + 1);
      }
    } else {
      throw std::runtime_error(
          "LR schedule is not supported, check train_lr_schedule flag possible values");
    }
  }
  optimizer_->setLr(lr);
}

void Trainer::reduceGrads() {
  collectParameters();
  if (reducer_) {
    for (auto& p : parameters_) {
      if (!p.isGradAvailable()) {
        p.addGrad(fl::constant(0.0, p.dims(), p.type(), false));
      }
      auto& grad = p.grad().array();
      p.grad().array() = grad;
      reducer_->add(p.grad());
    }
    reducer_->finalize();
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

bool Trainer::isMaster() const {
  return fl::getWorldRank() == 0;
}

void Trainer::checkArgs() const {
  if (version_ != FL_APP_LM_VERSION) {
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
  trainLossMeter_.reset();
  validLossMeter_.reset();
  runTimeMeter_.reset();
  batchTimerMeter_.reset();
  sampleTimerMeter_.reset();
  fwdTimeMeter_.reset();
  critFwdTimeMeter_.reset();
  bwdTimeMeter_.reset();
  optimTimeMeter_.reset();
  tokenCountMeter_.reset();
}

void Trainer::syncMeters() {
  syncMeter(trainLossMeter_);
  syncMeter(validLossMeter_);
  syncMeter(runTimeMeter_);
  syncMeter(batchTimerMeter_);
  syncMeter(sampleTimerMeter_);
  syncMeter(fwdTimeMeter_);
  syncMeter(critFwdTimeMeter_);
  syncMeter(bwdTimeMeter_);
  syncMeter(optimTimeMeter_);
  syncMeter(tokenCountMeter_);
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
  Serializer::save(
      path,
      FL_APP_LM_VERSION,
      network_,
      criterion_,
      optimizer_,
      epoch_,
      batchIdx_,
      gflagsStr_);

  if (!suffix.empty()) {
    Serializer::save(
        path + suffix,
        FL_APP_LM_VERSION,
        network_,
        criterion_,
        optimizer_,
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
  std::ostringstream oss;
  oss << "[epoch=" << epoch_ << " batch=" << batchIdx_ << "/"
      << trainDataset_->size() << "]";

  // Run time
  int rt = runTimeMeter_.value();
  oss << " | Run Time: "
      << format("%02d:%02d:%02d", (rt / 60 / 60), (rt / 60) % 60, rt % 60);
  oss << " | Batch Time(ms): "
      << format("%.2f", batchTimerMeter_.value() * 1000);
  oss << " | Sample Time(ms): "
      << format("%.2f", sampleTimerMeter_.value() * 1000);
  oss << " | Forward Time(ms): "
      << format("%.2f", fwdTimeMeter_.value() * 1000);
  oss << " | Criterion Forward Time(ms): "
      << format("%.2f", critFwdTimeMeter_.value() * 1000);
  oss << " | Backward Time(ms): "
      << format("%.2f", bwdTimeMeter_.value() * 1000);
  oss << " | Optimization Time(ms): "
      << format("%.2f", optimTimeMeter_.value() * 1000);

  oss << " | Throughput (Token/Sec): "
      << format(
             "%.2f",
             tokenCountMeter_.value()[0] * fl::getWorldSize() /
                 batchTimerMeter_.value());
  oss << " | Learning Rate " << format("%.6f", optimizer_->getLr());
  // Losses
  double loss = trainLossMeter_.value()[0];
  oss << " | Loss: " << format("%.2f", loss)
      << " PPL: " << format("%.2f", std::exp(loss));
  loss = validLossMeter_.value()[0];
  oss << " | Valid Loss: " << format("%.2f", loss)
      << " Valid PPL: " << format("%.2f", std::exp(loss));

  return oss.str();
}

} // namespace lm
} // namespace app
} // namespace fl
