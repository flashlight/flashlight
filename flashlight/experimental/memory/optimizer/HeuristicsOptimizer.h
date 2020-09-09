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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include "flashlight/flashlight/experimental/memory/AllocationLog.h"
#include "flashlight/flashlight/experimental/memory/allocator/ConfigurableMemoryAllocator.h"
#include "flashlight/flashlight/experimental/memory/allocator/MemoryAllocator.h"
#include "flashlight/flashlight/experimental/memory/optimizer/Optimizer.h"

namespace fl {

MemoryAllocatorConfiguration runHeuristicsOptimizer(
    const MemoryAllocator::Stats& stats,
    const MemoryOptimizerConfiguration& optimizerConfig,
    const MemoryAllocatorConfiguration& allocatorConfig,
    size_t numberOfIyterations);

std::vector<MemoryAllocatorConfiguration> heuristicsOptimizerSuggestions(
    const MemoryAllocator::Stats& stats,
    const MemoryOptimizerConfiguration& optimizerConfig,
    const MemoryAllocatorConfiguration& allocatorConfig,
    size_t maxNumSuggestions);

struct HeuristicsOptimizerDiagnosis {
  MemoryAllocator::Stats stats;
  // Indexes of subArenaStats with the specified issues.
  std::vector<size_t> highAllocationRatio;
  std::vector<size_t> lowAllocationRatio;
  std::vector<size_t> zeroUsage;
  std::vector<size_t> highInternalFragmentation;
  std::vector<size_t> highExternalFragmentation;
  std::vector<size_t> poorPerformance;
};

HeuristicsOptimizerDiagnosis heuristicsOptimizerDiagnosis(
    const MemoryAllocator::Stats& stats,
    const MemoryOptimizerConfiguration& optimizerConfig,
    const MemoryAllocatorConfiguration& allocatorConfig);

std::vector<MemoryAllocatorConfiguration> heuristicsOptimizerApplyDiagnosis(
    const HeuristicsOptimizerDiagnosis& allocatorConfigDiagnosis,
    const MemoryAllocator::Stats& stats,
    const MemoryOptimizerConfiguration& optimizerConfig,
    const MemoryAllocatorConfiguration& allocatorConfig,
    size_t numberOfConfigsToGenerate);

// Returns true on success and sets optimizedConfig.
// Optimized config adds another sub arena at subArenaIndex by pushing the
// current subArenaIndex to the next spot. The maxAllocationSize of the new
// arena and the arenas before and after it must be at least one multiple of
// (1<<alignmentNumberOfBits) apart.
bool splitSubArenaToReduceExternalFrag(
    const MemoryAllocatorConfiguration& config,
    size_t subArenaIndex,
    size_t alignmentNumberOfBits,
    MemoryAllocatorConfiguration* optimizedConfig);

} // namespace fl
