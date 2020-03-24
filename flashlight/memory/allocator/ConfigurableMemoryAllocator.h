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

#include "flashlight/memory/allocator/MemoryAllocator.h"

#include <memory>
#include <string>
#include <vector>

namespace fl {

// Defines configuration for allocator that manages a portion of the full arena
// described in MemoryAllocatorConfiguration.
struct SubArenaConfiguration {
  std::string name;
  size_t blockSize;
  size_t maxAllocationSize;
  // The portion of the arena that is given to this allocator entry is the sum
  // of all relativeSize over this relativeSize. The exact size of the allocator
  // is rounded to the nearest alignment.
  double relativeSize;
};

// Define configuration composite memory allocator such that a large
// arena is broken to sub arenas, each managed by a separate allocator.
struct MemoryAllocatorConfiguration {
  void* arena;
  // alignment is specified in number of bits, such that allocated pointer's
  // value is:
  // ptr = arena + n * (1 << alignmentNumberOfBits)
  size_t alignmentNumberOfBits;
  size_t arenaSizeInBytes;
  std::vector<SubArenaConfiguration> subArenaConfiguration;
};

// Returns a composite allocator configured by config.
std::unique_ptr<MemoryAllocator> CreateMemoryAllocator(
    const MemoryAllocatorConfiguration& config);

} // namespace fl
