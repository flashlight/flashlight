/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/decoder/PlGenerator.h"

#include <algorithm>
#include <chrono>
#include <thread>

#include "flashlight/pkg/runtime/common/SequentialBuilder.h"
#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/decoder/TranscriptionUtils.h"
#include "flashlight/pkg/speech/runtime/Helpers.h"

namespace {
constexpr const char* kPlDir = "generated_pl/";
constexpr const char* kPlSubdirPrefix = "epoch_";
} // namespace

using namespace fl::pkg::runtime;
using namespace fl::lib;
using namespace fl::lib::text;

namespace fl::pkg::speech {

PlGenerator::PlGenerator(
    const Dictionary& tokenDict,
    const fs::path& runPath,
    int worldRank,
    int worldSize,
    int batchSize,
    const fs::path& trainUnsupDir,
    const std::string& trainUnsupLists,
    const std::string& plEpoch,
    const std::string& plRatio,
    bool useExistingPl,
    float seedModelWER,
    double minInputSize,
    double maxInputSize,
    int minTargetSize,
    int maxTargetSize,
    const std::tuple<int, int, int>& padVal,
    fl::Dataset::DataTransformFunction inputTransform,
    fl::Dataset::DataTransformFunction targetTransform,
    fl::Dataset::DataTransformFunction wordTransform,
    TokenToWordFunc tokenToWord)
    : worldRank_(worldRank),
      isMaster_(worldRank_ == 0),
      worldSize_(worldSize),
      batchSize_(batchSize),
      tokenDict_(tokenDict),
      plDir_(runPath / kPlDir),
      useExistingPl_(useExistingPl),
      seedModelWER_(seedModelWER),
      minInputSize_(minInputSize),
      maxInputSize_(maxInputSize),
      minTargetSize_(minTargetSize),
      maxTargetSize_(maxTargetSize),
      padVal_(padVal),
      inputTransform_(inputTransform),
      targetTransform_(targetTransform),
      wordTransform_(wordTransform),
      tokenToWord_(tokenToWord) {
  // 1. Load PL generating intervals
  auto plEpochVec = lib::split(',', plEpoch, true);
  auto plRatioVec = lib::split(',', plRatio, true);

  if (plEpochVec.size() != plRatioVec.size()) {
    throw std::invalid_argument(
        "[PlGenerator] Size mismatch between pl_epoch and pl_ratio.");
  }

  plEpochs_.resize(plEpochVec.size());
  for (int i = 0; i < plEpochVec.size(); i++) {
    plEpochs_[i] = stoi(plEpochVec[i]);
  }

  for (int i = 0; i < plEpochVec.size(); i++) {
    auto ratio = stof(plRatioVec[i]);
    if (ratio < 0 || ratio > 1) {
      throw std::invalid_argument(
          "[PlGenerator] The value of pl_ratio should be in [0, 1].");
    }
    if (i > 0 && plEpochs_[i] <= plEpochs_[i - 1]) {
      throw std::invalid_argument(
          "[PlGenerator] Elements in pl_epoch should be in ascendant order.");
    }
    plUpdateMap_[plEpochs_[i]] = ratio;
  }

  // 2. Build the full unlabeled set
  std::vector<std::shared_ptr<const fl::Dataset>> allListDs;
  auto paths = lib::split(',', trainUnsupLists, true);
  for (auto& path : paths) {
    auto curListDs = std::make_shared<ListFileDataset>(
        trainUnsupDir / path,
        inputTransform_,
        targetTransform_,
        wordTransform_);

    allListDs.emplace_back(curListDs);
  }
  if (!allListDs.empty()) {
    if (isMaster_) {
      fs::create_directory(plDir_);
    }
    fullUnsupDs_ = std::make_shared<fl::ConcatDataset>(allListDs);
  }
}

std::string PlGenerator::reloadPl(int curEpoch) const {
  int lastPlEpoch = findLastPlEpoch(curEpoch);
  if (lastPlEpoch < 0) {
    return "";
  }

  fs::path plDir = plDir_ / (kPlSubdirPrefix + std::to_string(lastPlEpoch));

  bool isPLReady = true;
  for (int i = 0; i < worldSize_; i++) {
    auto listFinishPath = plDir / (std::to_string(i) + ".fns");
    if (!fs::exists(listFinishPath)) {
      isPLReady = false;
      break;
    }
  }
  if (isPLReady) {
    logMaster("[PlGenerator] Loading existing PL from " + plDir.string());
    return plDir;
  } else {
    logMaster("[PlGenerator] Failed to load PL from " + plDir.string());
    return "";
  }
}

std::string PlGenerator::regeneratePl(
    int curEpoch,
    const std::shared_ptr<fl::Module>& ntwrk,
    const std::shared_ptr<SequenceCriterion> criterion,
    const bool usePlugin /* = false */) const {
  if (plUpdateMap_.find(curEpoch) == plUpdateMap_.end()) {
    return "";
  }
  if (!fullUnsupDs_) {
    throw std::runtime_error("No unlabeled data is provided");
  }

  logMaster(
      "[PlGenerator] Regenerating PL at epoch " + std::to_string(curEpoch));
  fs::path plDir = plDir_ / (kPlSubdirPrefix + std::to_string(curEpoch));

  /* 0. Create logging folder */
  try {
    fs::create_directory(plDir);
  } catch (...) {
    // Pass. Allowing attempts from all processes to create the folder.
  }

  if (!fs::is_directory(plDir)) {
    throw std::runtime_error(
        "[PlGenerator] Failed to create " + plDir.string());
  }

  /* 1. select data */
  // shuffle
  auto ds1 = std::make_shared<fl::ShuffleDataset>(fullUnsupDs_, curEpoch);

  // select
  float ratio = plUpdateMap_.at(curEpoch);
  int nSelectedSamples = int(fullUnsupDs_->size() * ratio);
  std::vector<int64_t> sortedIds(nSelectedSamples);
  std::iota(sortedIds.begin(), sortedIds.end(), 0);
  auto ds2 = std::make_shared<fl::ResampleDataset>(ds1, sortedIds);

  // dispatch
  auto partitions =
      fl::partitionByRoundRobin(ds2->size(), worldRank_, worldSize_, 1);
  auto ds3 = std::make_shared<fl::ResampleDataset>(ds2, partitions);

  // prefetch
  auto selectedDs = std::make_shared<fl::PrefetchDataset>(ds3, 3, 3);

  logMaster(
      "[PlGenerator] " + std::to_string(nSelectedSamples) + "/" +
      std::to_string(fullUnsupDs_->size()) + " samples selected");

  /* 2. pseudo label generation */
  ntwrk->eval();
  auto newPlFile = plDir / (std::to_string(worldRank_) + ".lst");
  std::ofstream plStream(newPlFile);
  for (auto& sample : *selectedDs) {
    auto duration = sample[kDurationIdx].scalar<float>();
    if (duration < minInputSize_ || duration > maxInputSize_) {
      continue;
    }

    std::vector<std::string> words;
    if (useExistingPl_ && seedModelWER_ < currentModelWER_) {
      auto tokenTarget = sample[kTargetIdx].toHostVector<int>();
      words = tokenToWord_(tokenTarget, tokenDict_, false);
    } else {
      fl::Variable rawEmission;
      if (usePlugin) {
        rawEmission = ntwrk
                          ->forward(
                              {fl::input(sample[kInputIdx]),
                               fl::noGrad(sample[kDurationIdx])})
                          .front();
      } else {
        rawEmission = fl::pkg::runtime::forwardSequentialModuleWithPadMask(
            fl::input(sample[kInputIdx]), ntwrk, sample[kDurationIdx]);
      }
      auto tokenPrediction =
          criterion->viterbiPath(rawEmission.tensor()).toHostVector<int>();
      words = tokenToWord_(tokenPrediction, tokenDict_, true);
    }
    if (words.size() < minTargetSize_ || words.size() > maxTargetSize_) {
      continue;
    }

    auto sampleId = readSampleIds(sample[kSampleIdx]).front();
    auto inputPath = readSampleIds(sample[kPathIdx]).front();
    plStream << sampleId << "\t" << inputPath << "\t"
             << std::to_string(duration) << "\t" << lib::join(" ", words)
             << std::endl;
  }
  plStream.close();

  fs::path finishPlFile = plDir / (std::to_string(worldRank_) + ".fns");
  std::ofstream fnsStream(finishPlFile);
  fnsStream << "done";
  fnsStream.close();

  /* 3. waiting for all the other processes */
  fl::barrier();
  return plDir;
}

std::shared_ptr<fl::Dataset> PlGenerator::createTrainSet(
    const fs::path& trainDir,
    const fs::path& trainLists,
    const fs::path& trainUnsupDir,
    const std::string& batchingStrategy /* = kBatchStrategyNone */,
    int maxDurationPerBatch /* = 0 */) const {
  std::vector<fs::path> files;
  for (const auto& file : lib::split(",", trainLists, true)) {
    files.emplace_back(trainDir / file);
  }
  for (int i = 0; i < worldSize_; i++) {
    files.emplace_back(trainUnsupDir / (std::to_string(i) + ".lst"));
  }

  return createDataset(
      files,
      "",
      batchSize_,
      inputTransform_,
      targetTransform_,
      wordTransform_,
      padVal_,
      worldRank_,
      worldSize_,
      false, // allowEmpty
      batchingStrategy,
      maxDurationPerBatch);
}

void PlGenerator::setModelWER(const float& wer) {
  currentModelWER_ = wer;
}

int PlGenerator::findLastPlEpoch(int curEpoch) const {
  int lastPlEpoch = -1;
  for (const auto& i : plEpochs_) {
    if (i > curEpoch) {
      break;
    }
    lastPlEpoch = i;
  }
  return lastPlEpoch;
}

void PlGenerator::logMaster(const std::string& message) const {
  if (worldRank_ != 0) {
    return;
  }
  std::cerr << message << std::endl;
}

} // namespace fl
