/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * The configurable memory allocator is obtained by calling:
 * std::unique_ptr<MemoryAllocator> CreateMemoryAllocator(config)
 * Config defines a a set of allocators assembled in a CompositeMemoryAllocator.
 */

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include "flashlight/experimental/memory/AllocationLog.h"
#include "flashlight/experimental/memory/allocator/ConfigurableMemoryAllocator.h"
#include "flashlight/experimental/memory/allocator/MemoryAllocator.h"
#include "flashlight/experimental/memory/optimizer/Simulator.h"

namespace fl {

// Hyperparameters specifying loss function weights assigned for metrics and
// threshold for considering these metrics.
struct MemoryOptimizerConfiguration {
  double oomEventCountWeight = 0;
  double maxInternalFragmentationScoreWeight = 0;
  double maxInternalFragmentationScoreThreshold = 0;
  double maxExternalFragmentationScoreWeight = 0;
  double maxExternalFragmentationScoreThreshold = 0;
  double allocCountWeight = 0;
  double statsInBlocksMaxAllocatedRatioWeight = 0;
  double statsInBlocksMaxAllocatedRatioThreashold = 0;
  double performanceCostWeight = 0;
  // At the beginning of an iteration, we create a list of the configs with
  // lowest loss value. We choose at most beamSize configs. Around each of the
  // configs in that list we generate
  // numberOfConfigsToGeneratePerIteration/beamSize near configs.
  size_t beamSize;
  size_t numberOfConfigsToGeneratePerIteration;
  // Number of times that we create near configs, score them and choose the best
  // ones. Should be >=1 to have a least one scoring. Growing this number can be
  // very expensive.
  size_t numberOfIterations = 3;
  // The search radius per-iteration decrease ratio
  // 0..1
  // 0.5 is a very agressive factor of convergence.
  // 0.9 is a very meek factor of convergence.
  double learningRate = 0;
  size_t memorySize;

  void normalizeWeights();

  std::string prettyString() const;

  static MemoryOptimizerConfiguration loadJSon(std::istream& streamToConfig);
  void saveJSon(std::ostream& saveConfigStream) const;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("oomEventCountWeight", oomEventCountWeight),
       cereal::make_nvp(
           "maxInternalFragmentationScoreWeight",
           maxInternalFragmentationScoreWeight),
       cereal::make_nvp(
           "maxInternalFragmentationScoreThreshold",
           maxInternalFragmentationScoreThreshold),
       cereal::make_nvp(
           "maxExternalFragmentationScoreWeight",
           maxExternalFragmentationScoreWeight),
       cereal::make_nvp(
           "maxExternalFragmentationScoreWeight",
           maxExternalFragmentationScoreWeight),
       cereal::make_nvp(
           "maxExternalFragmentationScoreThreshold",
           maxExternalFragmentationScoreThreshold),
       cereal::make_nvp(
           "statsInBlocksMaxAllocatedRatioWeight",
           statsInBlocksMaxAllocatedRatioWeight),
       cereal::make_nvp(
           "statsInBlocksMaxAllocatedRatioThreashold",
           statsInBlocksMaxAllocatedRatioThreashold),
       cereal::make_nvp("beamSize", beamSize),
       cereal::make_nvp("allocCountWeight", allocCountWeight),
       cereal::make_nvp("numberOfIterations", numberOfIterations),
       cereal::make_nvp(
           "numberOfConfigsToGeneratePerIteration",
           numberOfConfigsToGeneratePerIteration),
       cereal::make_nvp("learningRate", learningRate),
       cereal::make_nvp("memorySize", memorySize));
  }
};

double memoryAllocatorStatsLossFunction(
    bool success,
    const MemoryAllocator::Stats& stats,
    double timeElapsed,
    double numAllocRatio,
    const MemoryOptimizerConfiguration& config);

struct MemoryAllocatorConfigurationLossOrdering {
  MemoryAllocatorConfigurationLossOrdering()
      : loss(-1), index(-1), completedWithSuccess(false) {}

  double loss;
  std::unique_ptr<MemoryAllocator::Stats> stats;
  int index;
  bool completedWithSuccess;
};

std::vector<MemoryAllocatorConfigurationLossOrdering>
orderMemoryAllocatorConfigByLoss(
    BlockingThreadPool& threadPool,
    const std::vector<MemoryAllocatorConfiguration>& haystack,
    const std::vector<AllocationEvent>& allocationLog,
    const MemoryOptimizerConfiguration& optimizerConfig);

// radius- generate configuration with the given ratio 0..1 of change from
//   maximal valid change.
std::vector<MemoryAllocatorConfiguration> generateNearConfig(
    const MemoryAllocatorConfiguration& center,
    double radius,
    size_t count,
    size_t arenaSize);

MemoryAllocatorConfiguration randomNearSearchOptimizer(
    std::vector<MemoryAllocatorConfiguration> newCenters,
    const std::vector<AllocationEvent>& allocationLog,
    const MemoryOptimizerConfiguration& optimizerConfig);

} // namespace fl
