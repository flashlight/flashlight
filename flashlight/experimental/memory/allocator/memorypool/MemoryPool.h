/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "flashlight/flashlight/experimental/memory/allocator/MemoryAllocator.h"

namespace fl {

// Breaks a large memory arena_ to bloks of size blockSize_ and allocate them
// one at a time. Allocation and deallocation costs O(1).
class MemoryPool : public MemoryAllocator {
 public:
  MemoryPool(
      std::string name,
      void* arena,
      size_t arenaSizeInBytes,
      size_t blockSize,
      double allocatedRatioJitThreshold,
      int logLevel = 1);
  ~MemoryPool() override;

  void* allocate(size_t size) override;
  void free(void* ptr) override;

  Stats getStats() const override;
  bool jitTreeExceedsMemoryPressure(size_t bytes) override;
  size_t getAllocatedSizeInBytes(void* ptr) const override;

  std::string prettyString() const override;

 private:
  struct Block {
    size_t allocatedSize = 0;
    long indexOfNext = -1;
  };

  void* toPtr(size_t index) const;

  MemoryAllocator::Stats stats_;
  size_t freeListSize_;
  long freeList_; // Index of first free block.
  std::vector<Block> blocks_;
  std::vector<size_t> currentlyAlloctedInBytes_;
  double allocatedRatioJitThreshold_;
  bool isNullAllocator_;
};

} // namespace fl
