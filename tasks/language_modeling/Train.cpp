/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "data/BlobDataset.h"

#include "extensions/common/SequentialBuilder.h"
#include "extensions/common/Utils.h"
#include "libraries/common/String.h"
#include "libraries/common/System.h"
#include "libraries/language/dictionary/Dictionary.h"
#include "libraries/language/dictionary/Utils.h"

using namespace fl::ext;
using namespace fl::lib;
using namespace fl::task::lm;

#define LOG_MASTER(lvl) LOG_IF(lvl, (fl::getWorldRank() == 0))

namespace {
DEFINE_int64(
    adsm_input_size,
    0,
    "input size of AdaptiveSoftMax (i.e. output size of network)");
DEFINE_string(adsm_cutoffs, "", "cutoffs for AdaptiveSoftMax");
DEFINE_int64(num_labels, 0, "# of classes for target labels");
DEFINE_int64(tokens_per_sample, 1024, "# of tokens per sample");
DEFINE_int64(saveiters, 0, "save every # iterations");
DEFINE_string(sample_break_mode, "none", "none, eos");
DEFINE_string(dictionary, "", "path to dictionary");
DEFINE_bool(use_dynamic_batching, false, "if or not use dynamic batching");
}

/* =========== Stateless helper functions ============ */
void initArrayFire() {
  af::setSeed(FLAGS_seed);
  af::setFFTPlanCacheSize(FLAGS_fftcachesize);
}

std::vector<int> parseCutoffs() {
  std::vector<int> cutoffs;
  auto tokens = split(',', FLAGS_adsm_cutoffs, true);
  for (const auto& token : tokens) {
    cutoffs.push_back(std::stoi(trim(token)));
  }
  cutoffs.push_back(FLAGS_num_labels);
  for (int i = 0; i + 1 < cutoffs.size(); ++i) {
    if (cutoffs[i] >= cutoffs[i + 1]) {
      throw std::invalid_argument("cutoffs must be strictly ascending");
    }
  }
  return cutoffs;
}

std::string serializeGflags(const std::string& separator="\n") {
  std::stringstream serialized;
  std::vector<gflags::CommandLineFlagInfo> allFlags;
  gflags::GetAllFlags(&allFlags);
  std::string currVal;
  for (auto itr = allFlags.begin(); itr != allFlags.end(); ++itr) {
    gflags::GetCommandLineOption(itr->name.c_str(), &currVal);
    serialized << "--" << itr->name << "=" << currVal << separator;
  }
  return serialized.str();
}

bool isMaster() {
  return fl::getWorldRank() == 0;
}

/* =========== The LM Trainer ============ */
struct Trainer {
 public:
  Trainer(int argc, char** argv) {
    // Parse or load persistent states
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_enable_distributed) {
      initDistributed(
          FLAGS_world_rank, FLAGS_world_size, 8, FLAGS_rndv_filepath);
    }

    // TODO: parse flags from file
    auto lastCheckPoint = pathsConcat(FLAGS_rundir, "model.bin");
    if (fileExists(lastCheckPoint)) {
      LOG_MASTER(INFO) << "Loading model checkpoint from: " << lastCheckPoint;
      fl::load(lastCheckPoint, *this);
      gflags::ReadFlagsFromString(gflags, gflags::GetArgv0(), true);
    } else {
      LOG_MASTER(INFO) << "No existing checkpoint found, creating fresh model ";
      createNetwork();
      createCriterion();
    }
    // DEBUG: override existing flags from command line
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    gflags = serializeGflags();
    LOG_MASTER(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

    // Initialize ephemeral states
    initArrayFire();
    if (FLAGS_enable_distributed) {
      reducer = std::make_shared<fl::CoalescingReducer>(
          1.0 / fl::getWorldSize(), true, true);
    }
    if (isMaster()) {
      dirCreate(FLAGS_rundir);
      logWriter = createOutputStream(
          pathsConcat(FLAGS_rundir, "log"), std::ios_base::app);
    }
    createDictionary();
    createDatasets();
    createOptimizer();

    LOG_MASTER(INFO) << "network (" << numTotalParams(network)
                     << " params): " << network->prettyString();
    LOG_MASTER(INFO) << "criterion (" << numTotalParams(criterion)
                     << " params): " << criterion->prettyString();
    LOG_MASTER(INFO) << "optimizer: " << optimizer->prettyString();
  }

  void trainStep() {
    network->train();
    criterion->train();
    setLr();

    // 1. Sample
    sampleTimerMeter.resume();
    const auto& sample = trainDataset->get(batchIdx);
    sampleTimerMeter.stopAndIncUnit();

    // 2. Forward
    fwdTimeMeter.resume();
    auto input = fl::Variable(sample[0], false);
    auto output = network->forward({input}).front();
    af::sync();
    critFwdTimeMeter.resume();
    auto target = fl::Variable(sample[1], false);
    auto loss = criterion->forward({output, target}).front();
    af::sync();
    fwdTimeMeter.stopAndIncUnit();
    critFwdTimeMeter.stopAndIncUnit();

    auto numWords = af::count<int>(sample[1] != kPadIdx);
    // std::cout << "*** " << numWords << " , " << target.elements();
    auto weight = numWords /
        static_cast<double>(FLAGS_tokens_per_sample * FLAGS_batchsize);
    trainLossMeter.add(af::mean<double>(loss.array()) / numWords, weight);
    wordCountMeter.add(numWords);

    // 3. Backward
    bwdTimeMeter.resume();
    optimizer->zeroGrad();
    loss.backward();
    reduceGrads();
    af::sync();
    bwdTimeMeter.stopAndIncUnit();

    // 4. Optimization
    optimTimeMeter.resume();
    fl::clipGradNorm(parameters, FLAGS_maxgradnorm);
    optimizer->step();
    af::sync();
    optimTimeMeter.stopAndIncUnit();
  }

  void evalStep() {
    network->eval();
    criterion->eval();

    for (auto& sample : *validDataset) {
      auto input = fl::Variable(sample[0], false);
      auto output = network->forward({input}).front();
      auto target = fl::Variable(sample[1], false);
      auto loss = criterion->forward({output, target}).front();
      auto numWords = af::count<int>(sample[1] != kPadIdx);
      auto weight = numWords /
          static_cast<double>(FLAGS_tokens_per_sample * FLAGS_batchsize);
      validLossMeter.add(af::mean<double>(loss.array()) / numWords, weight);
    }
  }

  void runTraining() {
    LOG_MASTER(INFO) << "training started (epoch=" << epoch
                     << " batch=" << batchIdx << ")";

    // DEBUG: addGrad() crash
    network->train();
    criterion->train();
    network->eval();
    criterion->eval();

    fl::allReduceParameters(network);
    fl::allReduceParameters(criterion);

    // Run Train
    for (;;) {
      if (batchIdx && batchIdx % trainDataset->size() == 0) {
        stopTimers();
        ++epoch;
        trainDataset->shuffle(FLAGS_seed + epoch);
      }
      runTimeMeter.resume();
      batchTimerMeter.resume();
      trainStep();
      batchTimerMeter.incUnit();
      ++batchIdx;
      if (FLAGS_reportiters && batchIdx % FLAGS_reportiters == 0) {
        stopTimers();
        evalStep();
        auto outputStr = progressString();
        LOG_MASTER(INFO) << outputStr;
        logWriter << outputStr << "\n" << std::flush;
        resetMeters();
      }
      if (FLAGS_saveiters && batchIdx % FLAGS_saveiters == 0) {
        stopTimers();
        saveCheckpoint(FLAGS_itersave);
      }
    }
  }

  void saveCheckpoint(bool batchSave = false) {
    if (!isMaster()) {
      return;
    }

    auto modelPath = pathsConcat(FLAGS_rundir, "model.bin");
    LOG_MASTER(INFO) << "saving model checkpoint (epoch=" << epoch
                     << " batch=" << batchIdx << ") to: " << modelPath;
    fl::save(modelPath, *this);

    if (batchSave) {
      modelPath = pathsConcat(modelPath, std::to_string(batchIdx));
      LOG_MASTER(INFO) << "saving model checkpoint (epoch=" << epoch
                       << " batch=" << batchIdx << ") to: " << modelPath;
      fl::save(modelPath, *this);
    }
  }

  void setLr() {
    // TODO: Support different LR schedulers
    optimizer->setLr(FLAGS_lr * std::min(batchIdx / double(FLAGS_warmup), 1.0));
  }

 private:
  // Persistent states
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<fl::Module> criterion;
  int64_t epoch{1};
  int64_t batchIdx{0};
  std::string gflags;
  FL_SAVE_LOAD(network, criterion, epoch, batchIdx, gflags)

  // Ephemeral states
  Dictionary dictionary;
  std::shared_ptr<BlobDataset> trainDataset;
  std::shared_ptr<BlobDataset> validDataset;

  std::shared_ptr<fl::Reducer> reducer;
  std::shared_ptr<fl::FirstOrderOptimizer> optimizer;
  std::vector<fl::Variable> parameters;

  fl::AverageValueMeter trainLossMeter;
  fl::AverageValueMeter validLossMeter;
  fl::TimeMeter runTimeMeter;
  fl::TimeMeter batchTimerMeter{true};
  fl::TimeMeter sampleTimerMeter{true};
  fl::TimeMeter fwdTimeMeter{true};
  fl::TimeMeter critFwdTimeMeter{true};
  fl::TimeMeter bwdTimeMeter{true};
  fl::TimeMeter optimTimeMeter{true};
  fl::AverageValueMeter wordCountMeter;

  std::ofstream logWriter;

  void createNetwork() {
    network = buildSequentialModule(
        pathsConcat(FLAGS_archdir, FLAGS_arch), 0, FLAGS_num_labels);
  }

  void createCriterion() {
    // TODO: Support different criterions
    auto softmax = std::make_shared<fl::AdaptiveSoftMax>(
        FLAGS_adsm_input_size, parseCutoffs());
    criterion = std::make_shared<fl::AdaptiveSoftMaxLoss>(
        softmax, fl::ReduceMode::SUM, kPadIdx);
  }

  void createOptimizer() {
    parameters = network->params();
    auto critParams = criterion->params();
    parameters.insert(parameters.end(), critParams.begin(), critParams.end());

    // TODO: Support different optimizers
    optimizer = std::make_shared<fl::SGDOptimizer>(
        parameters, FLAGS_lr, FLAGS_momentum, FLAGS_weightdecay, true);
  }

  void createDictionary() {
    dictionary.addEntry("<FL_DICT>");
    dictionary.addEntry("<PAD>");
    dictionary.addEntry("<EOS>");
    dictionary.addEntry("<UNK>");

    dictionary.setDefaultIndex(kUnkIdx);

    auto stream = createInputStream(FLAGS_dictionary);
    std::string line;
    while (std::getline(stream, line)) {
      if (line.empty()) {
        continue;
      }
      auto tkns = splitOnWhitespace(line, true);
      dictionary.addEntry(tkns.front());
    }
    if (!dictionary.isContiguous()) {
      throw std::runtime_error("Invalid dictionary format - not contiguous");
    }
  }

  void createDatasets() {
    trainDataset = std::make_shared<BlobDataset>(
        FLAGS_datadir,
        FLAGS_train,
        dictionary,
        fl::getWorldRank(),
        fl::getWorldSize(),
        FLAGS_tokens_per_sample,
        FLAGS_batchsize,
        kPadIdx,
        kEosIdx,
        FLAGS_sample_break_mode,
        FLAGS_use_dynamic_batching);

    validDataset = std::make_shared<BlobDataset>(
        FLAGS_datadir,
        FLAGS_valid,
        dictionary,
        fl::getWorldRank(),
        fl::getWorldSize(),
        FLAGS_tokens_per_sample,
        FLAGS_batchsize,
        kPadIdx,
        kEosIdx,
        FLAGS_sample_break_mode,
        FLAGS_use_dynamic_batching);

    LOG_MASTER(INFO) << "train dataset: " << trainDataset->size() << " samples";
    LOG_MASTER(INFO) << "valid dataset: " << validDataset->size() << " samples";
  }

  void resetMeters() {
    trainLossMeter.reset();
    validLossMeter.reset();
    runTimeMeter.reset();
    batchTimerMeter.reset();
    sampleTimerMeter.reset();
    fwdTimeMeter.reset();
    critFwdTimeMeter.reset();
    bwdTimeMeter.reset();
    optimTimeMeter.reset();
    wordCountMeter.reset();
  }

  void syncMeters() {
    syncMeter(trainLossMeter);
    syncMeter(validLossMeter);
    syncMeter(runTimeMeter);
    syncMeter(batchTimerMeter);
    syncMeter(sampleTimerMeter);
    syncMeter(fwdTimeMeter);
    syncMeter(critFwdTimeMeter);
    syncMeter(bwdTimeMeter);
    syncMeter(optimTimeMeter);
    syncMeter(wordCountMeter);
  }

  void stopTimers() {
    runTimeMeter.stop();
    batchTimerMeter.stop();
    sampleTimerMeter.stop();
    fwdTimeMeter.stop();
    critFwdTimeMeter.stop();
    bwdTimeMeter.stop();
    optimTimeMeter.stop();
  }

  std::string progressString() {
    std::ostringstream oss;
    oss << "[epoch=" << epoch << " batch=" << batchIdx << "/"
        << trainDataset->size() << "]";

    syncMeters();
    // Run time
    int rt = runTimeMeter.value();
    oss << " | Run Time: "
        << format("%02d:%02d:%02d", (rt / 60 / 60), (rt / 60) % 60, rt % 60);
    oss << " | Batch Time(ms): "
        << format("%.2f", batchTimerMeter.value() * 1000);
    oss << " | Sample Time(ms): "
        << format("%.2f", sampleTimerMeter.value() * 1000);
    oss << " | Forward Time(ms): "
        << format("%.2f", fwdTimeMeter.value() * 1000);
    oss << " | Criterion Forward Time(ms): "
        << format("%.2f", critFwdTimeMeter.value() * 1000);
    oss << " | Backward Time(ms): "
        << format("%.2f", bwdTimeMeter.value() * 1000);
    oss << " | Optimization Time(ms): "
        << format("%.2f", optimTimeMeter.value() * 1000);

    oss << " | Throughput (Word/Sec): "
        << format(
               "%.2f",
               wordCountMeter.value()[0] * fl::getWorldSize() /
                   batchTimerMeter.value());

    // Losses
    double loss = trainLossMeter.value()[0];
    oss << " | Loss: " << format("%.2f", loss)
        << " PPL: " << format("%.2f", std::exp(loss));
    loss = validLossMeter.value()[0];
    oss << " | Valid Loss: " << format("%.2f", loss)
        << " Valid PPL: " << format("%.2f", std::exp(loss));

    return oss.str();
  }

  void reduceGrads() {
    if (reducer) {
      for (auto& p : parameters) {
        if (!p.isGradAvailable()) {
          p.addGrad(fl::constant(0.0, p.dims(), p.type(), false));
        }
        reducer->add(p.grad());
      }
      reducer->finalize();
    }
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  auto trainer = Trainer(argc, argv);
  trainer.runTraining();
}