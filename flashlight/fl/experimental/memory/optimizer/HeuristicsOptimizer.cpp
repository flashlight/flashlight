/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/experimental/memory/optimizer/Optimizer.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/common/threadpool/ThreadPool.h"
#include "flashlight/fl/experimental/memory/allocator/ConfigurableMemoryAllocator.h"
#include "flashlight/fl/experimental/memory/optimizer/Simulator.h"

namespace fl {

double highUsageRelativeSizeRatio = 1.5;
// Sub arean size is decreased when MaxAllocatedRatio <
// statsInBlocksMaxAllocatedRatioThreashold
double statsInBlocksMaxAllocatedRatioThreashold = 0.3;

bool splitSubArenaToReduceExternalFrag(
    const MemoryAllocatorConfiguration& allocatorConfig,
    size_t subArenaIndex,
    size_t alignmentNumberOfBits_,
    MemoryAllocatorConfiguration* optimizedConfig) {
  if (subArenaIndex >= allocatorConfig.subArenaConfiguration_.size()) {
    std::stringstream ss;
    ss << "splitSubArena(allocatorConfig=" << allocatorConfig.prettyString()
       << " subArenaIndex=" << subArenaIndex
       << ") subArenaIndex must be < allocatorConfig.subArenaConfiguration_.size()="
       << allocatorConfig.subArenaConfiguration_.size();
    throw std::invalid_argument(ss.str());
  }

  // If this is the first sub arena set prevMaxAllocationSize=0
  const size_t prevMaxAllocationSize =
      ((subArenaIndex > 0)
           ? allocatorConfig.subArenaConfiguration_.at(subArenaIndex - 1)
                 .maxAllocationSize_
           : 0);
  const size_t nextMaxAllocationSize =
      allocatorConfig.subArenaConfiguration_.at(subArenaIndex)
          .maxAllocationSize_;
  const size_t maxAllocationSizeDiff =
      nextMaxAllocationSize - prevMaxAllocationSize;
  const size_t minAligmentInDiffCount =
      maxAllocationSizeDiff >> alignmentNumberOfBits_;

  if (minAligmentInDiffCount <= 1) {
    return false;
  }

  *optimizedConfig = allocatorConfig;
  // Split the relativeSize_ between the exiting and the new sub arenas.
  optimizedConfig->subArenaConfiguration_.at(subArenaIndex).relativeSize_ /=
      2.0;

  SubArenaConfiguration newSubArena =
      optimizedConfig->subArenaConfiguration_.at(subArenaIndex);

  // maxAllocationSize_ = aligned half way between prevMaxAllocationSize and
  // nextMaxAllocationSize.
  newSubArena.maxAllocationSize_ =
      prevMaxAllocationSize + (maxAllocationSizeDiff / 2);

  optimizedConfig->subArenaConfiguration_.insert(
      optimizedConfig->subArenaConfiguration_.begin() + subArenaIndex,
      std::move(newSubArena));
  return true;
}

std::vector<MemoryAllocatorConfiguration> heuristicsSuggestOptimization(
    const MemoryAllocator::Stats& stats,
    const MemoryOptimizerConfiguration& optimizerConfig,
    const MemoryAllocatorConfiguration& allocatorConfig,
    size_t numberOfConfigsToGenerate) {
  std::vector<MemoryAllocatorConfiguration> suggestions;

  // Iterate from the most significant to the least significant arena.
  for (int i = stats.subArenaStats.size() - 1; i >= 0; --i) {
    const MemoryAllocator::Stats& subArenaStats = stats.subArenaStats.at(i);

    // Handle high extrenal fragmentation.
    if (subArenaStats.maxExternalFragmentationScore >
        optimizerConfig.maxExternalFragmentationScoreThreshold) {
      MemoryAllocatorConfiguration suggested = allocatorConfig;
      if (splitSubArenaToReduceExternalFrag(
              allocatorConfig,
              i,
              allocatorConfig.alignmentNumberOfBits_,
              &suggested)) {
        FL_LOG(fl::INFO)
            << "heuristicsSuggestOptimization() Handle high extrenal fragmentation."
            << "\norig=" << allocatorConfig.prettyString()
            << "\nsuggested=" << suggested.prettyString();

        suggestions.push_back(std::move(suggested));

        if (suggestions.size() >= numberOfConfigsToGenerate) {
          return suggestions;
        }
      }
    }

    // Handle high internal fragmentation.
    if (subArenaStats.maxInternalFragmentationScore >
        optimizerConfig.maxInternalFragmentationScoreThreshold) {
      const int minBlockSizeCnt =
          allocatorConfig.subArenaConfiguration_.at(i).blockSize_ >>
          allocatorConfig.alignmentNumberOfBits_;
      if (minBlockSizeCnt > 1) {
        MemoryAllocatorConfiguration suggested = allocatorConfig;
        suggested.subArenaConfiguration_.at(i).blockSize_ =
            ((minBlockSizeCnt - 1) << allocatorConfig.alignmentNumberOfBits_);

        FL_LOG(fl::INFO)
            << "heuristicsSuggestOptimization() Handle high internal fragmentation."
            << "\norig=" << allocatorConfig.prettyString()
            << "\nsuggested=" << suggested.prettyString();

        suggestions.push_back(std::move(suggested));

        if (suggestions.size() >= numberOfConfigsToGenerate) {
          return suggestions;
        }
      }
    }

    // Handle high allocation ratio.
    if (subArenaStats.oomEventCount > 0 ||
        subArenaStats.statsInBlocks.maxAllocatedRatio() >
            optimizerConfig.statsInBlocksMaxAllocatedRatioThreashold) {
      MemoryAllocatorConfiguration suggested = allocatorConfig;
      suggested.subArenaConfiguration_.at(i).relativeSize_ *=
          highUsageRelativeSizeRatio;
      suggested.normalize();

      FL_LOG(fl::INFO)
          << "heuristicsSuggestOptimization() Handle high allocation ratio."
          << "\norig=" << allocatorConfig.prettyString()
          << "\nsuggested=" << suggested.prettyString();

      suggestions.push_back(std::move(suggested));

      if (suggestions.size() >= numberOfConfigsToGenerate) {
        return suggestions;
      }
    }

    // Handle low allocation ratio.
    if (subArenaStats.statsInBlocks.maxAllocatedRatio() <=
            optimizerConfig.statsInBlocksMaxAllocatedRatioThreashold &&
        allocatorConfig.subArenaConfiguration_.size() > 1) {
      // Remove allocators with zero peak usage.
      if (subArenaStats.statsInBlocks.maxAllocatedRatio() == 0) {
        MemoryAllocatorConfiguration suggested = allocatorConfig;

        suggested.subArenaConfiguration_.erase(
            suggested.subArenaConfiguration_.begin() + i);

        FL_LOG(fl::INFO)
            << "heuristicsSuggestOptimization() Handle zero allocation ratio."
            << "\norig=" << allocatorConfig.prettyString()
            << "\nsuggested=" << suggested.prettyString();

        suggestions.push_back(std::move(suggested));

        if (suggestions.size() >= numberOfConfigsToGenerate) {
          return suggestions;
        }
      }
    }
  }

  return suggestions;
};

}; // namespace fl
