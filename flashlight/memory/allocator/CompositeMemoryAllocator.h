/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "flashlight/memory/allocator/MemoryAllocator.h"

namespace fl {

// CompositeMemoryAllocator reduces external fragmentation by grouping similar
// size objects to the same allocator. It is a collection of simpler memory
// allocators ordered by maximum allocation size, from small to large.
// Allocation requests are delegated to the allocatos whos maximum allocation
// size is the nearest larger value. If that allocator fails to satisfy the
// allocation, then the request is delegated to the next one up, and so on.
class CompositeMemoryAllocator : public MemoryAllocator {
 public:
  CompositeMemoryAllocator();
  ~CompositeMemoryAllocator() override = default;

  struct AllocatorAndCriteria {
    size_t maxAllocationSize;
    std::unique_ptr<MemoryAllocator> allocator;

    // Support for std::sort() of AllocatorAndCriteria collection.
    bool operator<(const AllocatorAndCriteria& other) const {
      return (maxAllocationSize < other.maxAllocationSize);
    }

    // Dumps prettyString() of self followed by prettyString() of internal
    // allocators. Example with 3 freelist allocators:
    // CompositeMemoryAllocator{totalNumberOfAllocations_=9 numberOfAllocators=3
    // allcatorsAndCriterias={CompositeMemoryAllocator{ maxAllocationSize=4
    // allocator=FreeList{name_=small-freelist stats=Stats{arena=0x10000
    // blockSize=2 allocationsCount=3 deAllocationsCount=2
    // internalFragmentationScore=0 externalFragmentationScore=0.5
    // statsInBytes={arenaSize=20 freeCount=16 allocatedCount=4
    // allocatedRatio=0.2}} statsInBlocks={arenaSize=10 freeCount=8
    // allocatedCount=2 allocatedRatio=0.2}}} largestChunkInBlocks_=4
    // totalNumberOfChunks_=3 numberOfAllocatedChunks=1 freeListSize_=2
    // freeList_={Chunk{blockStartIndex_=0 sizeInBlocks_=4 next_=0x7f02dda67330
    // prev_=0} Chunk{blockStartIndex_=6 sizeInBlocks_=4 next_=0
    // prev_=0x7f02dda67c60} }CompositeMemoryAllocator{ maxAllocationSize=20
    // allocator=FreeList{name_=medium-freelist stats=Stats{arena=0x20000
    // blockSize=2 allocationsCount=3 deAllocationsCount=2
    // internalFragmentationScore=0 externalFragmentationScore=0.5
    // statsInBytes={arenaSize=100 freeCount=80 allocatedCount=20
    // allocatedRatio=0.2}} statsInBlocks={arenaSize=50 freeCount=40
    // allocatedCount=10 allocatedRatio=0.2}}} largestChunkInBlocks_=20
    // totalNumberOfChunks_=3 numberOfAllocatedChunks=1 freeListSize_=2
    // freeList_={Chunk{blockStartIndex_=0 sizeInBlocks_=20 next_=0x7f02dda67fc0
    // prev_=0} Chunk{blockStartIndex_=30 sizeInBlocks_=20 next_=0
    // prev_=0x7f02dda67000} }CompositeMemoryAllocator{
    // maxAllocationSize=18446744073709551615
    // allocator=FreeList{name_=large-freelist stats=Stats{arena=0x30000
    // blockSize=2 allocationsCount=3 deAllocationsCount=2
    // internalFragmentationScore=0 externalFragmentationScore=0.5
    // statsInBytes={arenaSize=200 freeCount=160 allocatedCount=40
    // allocatedRatio=0.2}} statsInBlocks={arenaSize=100 freeCount=80
    // allocatedCount=20 allocatedRatio=0.2}}} largestChunkInBlocks_=40
    // totalNumberOfChunks_=3 numberOfAllocatedChunks=1 freeListSize_=2
    // freeList_={Chunk{blockStartIndex_=0 sizeInBlocks_=40 next_=0x7f02de950160
    // prev_=0} Chunk{blockStartIndex_=60 sizeInBlocks_=40 next_=0
    // prev_=0x7f02de950730} }
    std::string prettyString() const;
  };

  void* allocate(size_t size) override;
  void free(void* ptr) override;

  // Retunes weighted (by relative memory size) summary of internal allocators
  // stats.
  Stats getStats() const override;
  size_t getAllocatedSizeInBytes(void* ptr) const override;

  std::string prettyString() const;

  void add(AllocatorAndCriteria allocatorAndCriteria);

 private:
  // Allocation used for keeping track of allocated memory objects. It is
  // used by free() to know to which allocator we want to return that memory
  // and for stats calculations.
  struct Allocation {
    size_t size;
    size_t allocatorsAndCriteriasIndex;
  };

  std::vector<AllocatorAndCriteria> allocatorsAndCriterias_;
  std::unordered_map<void*, Allocation> ptrToAllocation_;
  size_t totalNumberOfAllocations_;
  size_t arenaSizeInBlocks_;
  size_t arenaSizeInBytes_;
};

} // namespace fl
