/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <string>

namespace fl {

// Memory allocator base class. Supports allocation, deallocation, stats
// collections, and giving information about allocated pointers.
class MemoryAllocator {
 public:
  MemoryAllocator(std::string name);
  virtual ~MemoryAllocator() = default;

  virtual void* allocate(size_t size) = 0;
  virtual void free(void* ptr) = 0;

  // Helper for Stats
  struct CommonStats {
    size_t arenaSize;
    size_t freeCount;
    size_t allocatedCount;
    double allocatedRatio; // allocatedCount/arenaSize

    std::string prettyString() const;
  };

  struct Stats {
    void* arena;
    size_t blockSize;
    size_t allocationsCount;
    size_t deAllocationsCount;
    double internalFragmentationScore; // 0..1  no-frag .. highly-frag
    double externalFragmentationScore; // 0..1  no-frag .. highly-frag

    CommonStats statsInBytes;
    CommonStats statsInBlocks;

    std::string prettyString() const;
  };
  virtual Stats getStats() const = 0;

  virtual size_t getAllocatedSizeInBytes(void* ptr) const = 0;

  virtual std::string prettyString() const = 0;

  const std::string& getName() const;

 protected:
  const std::string name_;
};
} // namespace fl
