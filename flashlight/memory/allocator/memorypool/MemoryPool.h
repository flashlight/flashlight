/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "flashlight/memory/allocator/MemoryAllocator.h"

namespace fl {

// Breaks a large memory arena_ to bloks of size blockSize_ and allocate them
// one at a time. Allocation and deallocation costs O(1).
class MemoryPool : public MemoryAllocator {
 public:
  MemoryPool(std::string name, void* arena, size_t arenaSizeInBytes, size_t blockSize);
  ~MemoryPool();

  void* allocate(size_t size) override;
  void free(void* ptr) override;

  Stats getStats() const override;

  size_t getAllocatedSizeInBytes(void* ptr) const override;

  std::string prettyString() const override;

 private:
   struct Block {
     size_t allocatedSize = 0;
     long indexOfNext = -1;
   };

  void* toPtr(size_t index) const ;

  const std::string name_;
  void* const arena_;
  const size_t arenaSizeInBytes_;
  const size_t arenaSizeInBlocks_;
  const size_t blockSize_;
  size_t allocatedBytesCount_;
  size_t allocatedBlocksCount_;
  size_t totalNumberOfAllocations_;
  size_t totalNumberOfFree_;
  size_t freeListSize_;
  long freeList_; // Index of first free block.
  std::vector<Block> blocks_;
};

} // namespace fl
