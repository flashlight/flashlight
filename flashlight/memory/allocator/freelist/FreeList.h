/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "flashlight/memory/allocator/MemoryAllocator.h"

namespace fl {

// Breaks a large memory @arena to blocks of size @blockSize and allocate them
// in chunks. Each chunk contains one or more continuous blocks. Chunk is the
// datastructure used to keep track of blocks kept internally in a free list and
// of blocks allocated to the user. Currently allocation and deallocation cost
// O(n) where 'n' is the size of the freelist. The time constant per element in
// 'n', seems to be too small to mandate using a more complex log(n) data
// structure.
class FreeList : public MemoryAllocator {
 public:
  FreeList(
      std::string name,
      void* arena,
      size_t arenaSizeInBytes,
      size_t blockSize);
  ~FreeList() override;

  void* allocate(size_t size) override;
  void free(void* ptr) override;

  Stats getStats() const override;
  size_t getAllocatedSizeInBytes(void* ptr) const override;

  std::string prettyString() const override;

 private:
  // Descriptor of a memory region of continuous blocks.
  struct Chunk {
    Chunk(size_t blockStartIndex, size_t sizeInBlocks);

    // Output example:
    // FreeList{name_=NormalDistribution stats=Stats{arena=0x1000 blockSize=3
    // allocationsCount=100000(10k) deAllocationsCount=80000
    // internalFragmentationScore=0.00199318 externalFragmentationScore=0.106018
    // statsInBytes={arenaSize=50000000(47MB) freeCount=40004810(38MB)
    // allocatedCount=9995190(9MB) allocatedRatio=0.199904}}
    // statsInBlocks={arenaSize=16666666(15MB) freeCount=13328282(12MB)
    // allocatedCount=3338384(3260KB) allocatedRatio=0.200303}}}
    // largestChunkInBlocks_=11915249 totalNumberOfChunks_=27440
    // numberOfAllocatedChunks=20000 freeListSize_=7440
    // freeList_={Chunk{blockStartIndex_=0 sizeInBlocks_=309
    // next_=0x7f03e17aa070 prev_=0} Chunk{blockStartIndex_=495
    // sizeInBlocks_=211 next_=0x7f03e17ac0b0 prev_=0x7f03e174fe60}
    // Chunk{blockStartIndex_=865 sizeInBlocks_=179 next_=0x7f03e08988d0
    // prev_=0x7f03e17aa070} ... not showing the next_ 7437 chunks.}
    std::string prettyString() const;

    size_t blockStartIndex_ = 0;
    size_t sizeInBytes_ = 0;
    size_t sizeInBlocks_ = 0;
    Chunk* next_;
    Chunk* prev_;
  };

  Chunk* findBestFit(size_t nBlocksNeed);

  void chopChunkIfNeeded(Chunk* chunk, size_t nBlocksNeed);

  void addToFreeList(Chunk* chunk);

  Chunk* mergeWithPrevNeighbors(Chunk* chunk);
  void mergeWithNextNeighbors(Chunk* chunk);
  void mergeWithNeighbors(Chunk* chunk);

  void removeFromFreeList(Chunk* chunk);

  void* memoryAddress(void* ptr, size_t numBlocksOffset) const {
    return reinterpret_cast<char*>(ptr) + numBlocksOffset * blockSize_;
  }

  Chunk* getChunkAndRemoveFromMap(
      void* ptr,
      const std::string& callerNameForErrorMessage);
  const Chunk* getChunk(void* ptr, const std::string& callerNameForErrorMessage)
      const;

  void* const arena_;
  const size_t arenaSizeInBytes_;
  const size_t arenaSizeInBlocks_;
  const size_t blockSize_;
  size_t allocatedBytesCount_;
  size_t allocatedBlocksCount_;
  // mutable since largestChunkInBlocks_ is recalculated by getStats() which is
  // const.
  mutable size_t largestChunkInBlocks_;
  size_t totalNumberOfChunks_;
  size_t totalNumberOfAllocations_;
  size_t totalNumberOfFree_;
  size_t freeListSize_;
  // Freelist is sorted by block index from small to large.
  Chunk* freeList_;
  // Mapping from pointer of memory allocated to the user to Chunk data
  // structure describing that memory chunk.
  std::unordered_map<void*, Chunk*> pointerToAllocatedChunkMap_;
  // Vector of the block size and relative order of all allocations requests.
  // Used for stats analysis.
  std::vector<size_t> historyOfAllocationRequestInBlocks_;
};

} // namespace fl
