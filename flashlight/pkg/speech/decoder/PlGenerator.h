/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/criterion/criterion.h"
#include "flashlight/fl/contrib/contrib.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"

namespace fl {
namespace pkg {
namespace speech {

using TokenToWordFunc = std::function<std::vector<
    std::string>(const std::vector<int>&, const lib::text::Dictionary&, bool)>;

/**
 * PlGenerator is an easy plug-in to Train.cpp for generating pseudo labels on
 * the fly. It is flag-independent.
 *
 * References:
 *  - IPL: https://arxiv.org/abs/2005.09267
 *  - slimIPL: https://arxiv.org/abs/2010.11524
 *
 * Sample usage in Train.cpp:
 *  // Initialize
 *  plGen = PlGenerator(...);
 *
 *  // Load existing pseudo labels before training starts
 *  unsupDataDir = plGen.reloadPl(current_epoch);
 *  trainset = plGen.createTrainSet(
 *    supDataDir,
 *    supTrainLists,
 *    unsupDataDir);
 *
 *  main train loop {
 *    // Main train logic with `trainset` for current epoch
 *
 *    current_epoch++;
 *
 *    // Try regenerate pseudo labels with the current model
 *    unsupDataDir = plGen.reloadPl(current_epoch, model);
 *    trainset = plGen.createTrainSet(
 *      supDataDir,
 *      supTrainLists,
 *      unsupDataDir);
 *  }
 *
 */
class PlGenerator {
 public:
  PlGenerator(
      const lib::text::Dictionary& tokenDict,
      const std::string& runPath,
      int worldRank,
      int worldSize,
      int batchSize,
      const std::string& trainUnsupDir,
      const std::string& trainUnsupLists,
      const std::string& plEpoch,
      const std::string& plRatio,
      bool useExistingPl,
      float seedModelWER,
      double minInputSize, // in milliseconds
      double maxInputSize, // in milliseconds
      int minTargetSize, // in words
      int maxTargetSize, // in words
      const std::tuple<int, int, int>& padVal,
      fl::Dataset::DataTransformFunction inputTransform,
      fl::Dataset::DataTransformFunction targetTransform,
      fl::Dataset::DataTransformFunction wordTransform,
      TokenToWordFunc tokenToWord);

  /*
   * To resume trainig, try to load existing pseudo labels.
   * `nullptr` is returned if loading fails.
   */
  std::string reloadPl(int curEpoch) const;

  /*
   * To regenerate pseudo labels with the current model.
   * `nullptr` is returned if it's not supposed to do relabeling at the current
   * epoch.
   */
  std::string regeneratePl(
      int curEpoch,
      const std::shared_ptr<fl::Module>& ntwrk,
      const std::shared_ptr<SequenceCriterion> criterion,
      const bool usePlugin = false) const;

  /*
   * This function will create a mixture of supervised data and unalabeled data
   * with pseudo labels.
   */
  std::shared_ptr<fl::Dataset> createTrainSet(
      const std::string& trainDir,
      const std::string& trainLists,
      const std::string& trainUnsupDir,
      const std::string& batchingStrategy = kBatchStrategyNone,
      int maxDurationPerBatch = 0) const;

  /* To set the WER of current model in PlGenerator */
  void setModelWER(const float& wer);

 private:
  int worldRank_;
  bool isMaster_;
  int worldSize_;
  int batchSize_;

  lib::text::Dictionary tokenDict_;
  std::string plDir_;

  bool useExistingPl_;
  double seedModelWER_;
  double currentModelWER_;

  float minInputSize_;
  float maxInputSize_;
  int minTargetSize_;
  int maxTargetSize_;

  std::tuple<int, int, int> padVal_;
  fl::Dataset::DataTransformFunction inputTransform_;
  fl::Dataset::DataTransformFunction targetTransform_;
  fl::Dataset::DataTransformFunction wordTransform_;
  TokenToWordFunc tokenToWord_;

  std::shared_ptr<fl::Dataset> fullUnsupDs_;
  std::vector<int> plEpochs_;
  std::unordered_map<int, float> plUpdateMap_;

  int findLastPlEpoch(int curEpoch) const;
  void logMaster(const std::string& message) const;
};

} // namespace speech
} // namespace pkg
} // namespace fl
