/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "flashlight/memory/allocator/memorypool/MemoryPool.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "flashlight/common/Logging.h"

namespace fl {

MemoryPool::MemoryPool(
    std::string name,
    void* arena,
    size_t arenaSizeInBytes,
    size_t blockSize)
    : name_(std::move(name)),
      arena_(arena),
      arenaSizeInBytes_(arenaSizeInBytes),
      arenaSizeInBlocks_(arenaSizeInBytes_ / blockSize),
      blockSize_(blockSize),
      allocatedBytesCount_(0),
      allocatedBlocksCount_(0),
      totalNumberOfAllocations_(0),
      totalNumberOfFree_(0),
      blocks_(arenaSizeInBlocks_) {
  // point to block index 0
  freeList_ = 0;
  // Each block point to the next.
  for (size_t i = 0; i < (arenaSizeInBlocks_ - 1); ++i) {
    blocks_[i].indexOfNext = i + 1;
  }
  // last block points to -1.
  blocks_[arenaSizeInBlocks_ - 1].indexOfNext = -1;
  LOG(INFO) << "Created " << prettyString();
}

MemoryPool::~MemoryPool() {
  if (allocatedBlocksCount_) {
    LOG(WARNING) << "~MemoryPool() there are still " << allocatedBlocksCount_
                 << " blocks_ held by the user";
  }
}

void* MemoryPool::toPtr(size_t index) const {
  if (index < 0) {
    return nullptr;
  }
  return static_cast<char*>(arena_) + index * blockSize_;
}

void* MemoryPool::allocate(size_t size) {
  if (size == 0) {
    return nullptr;
  }
  if (size > blockSize_) {
    std::stringstream stream;
    stream << "allocate(size=" << size
           << ") must be <= blockSize_=" << blockSize_;
    throw std::invalid_argument(stream.str());
  }
  if (freeList_ < 0) {
    std::stringstream stream;
    stream << "allocate(size=" << size << ") OOM (out of memory)."
           << prettyString();
    throw std::invalid_argument(stream.str());
  }

  long blockIndex = freeList_;
  Block& block = blocks_[blockIndex];
  block.allocatedSize = size;
  freeList_ = block.indexOfNext;
  block.indexOfNext = -1;

  // stats
  ++totalNumberOfAllocations_;
  allocatedBytesCount_ += size;
  ++allocatedBlocksCount_;
  --freeListSize_;

  VLOG(2) << " MemoryPool::allocate(size=" << size
          << ") ptr=" << toPtr(blockIndex) << " blockIndex=" << blockIndex
          << " freeListSize_=" << freeListSize_;

  return toPtr(blockIndex);
}

void MemoryPool::free(void* ptr) {
  const size_t offset = (static_cast<char*>(ptr) - static_cast<char*>(arena_));
  const size_t blockIndex = offset / blockSize_;
  if (offset < 0 || blockIndex >= arenaSizeInBlocks_ || (offset % blockSize_)) {
    std::stringstream stream;
    stream << "MemoryPool::free(ptr=" << ptr
           << ") pointer is not managed by MemoryPool. offset=" << offset << " "
           << prettyString();
    throw std::invalid_argument(stream.str());
  }
  Block& block = blocks_[blockIndex];

  VLOG(2) << " MemoryPool::free(ptr=" << ptr << ") offset=" << offset
          << " blockIndex=" << blockIndex
          << " block.allocatedSize=" << block.allocatedSize;

  // stats
  allocatedBytesCount_ -= block.allocatedSize;
  --allocatedBlocksCount_;
  ++totalNumberOfFree_;

  block.indexOfNext = freeList_;
  block.allocatedSize = 0;
  freeList_ = blockIndex;
}

std::string MemoryPool::prettyString() const {
  const Stats stats = getStats();

  std::stringstream stream;
  stream << "MemoryPool{name_=" << name_ << " stats=" << stats.prettyString();
  stream << " freeListSize_=" << freeListSize_ << "}";

  return stream.str();
}

MemoryAllocator::Stats MemoryPool::getStats() const {
  Stats stats;
  stats.statsInBytes.arenaSize = arenaSizeInBytes_;
  stats.statsInBytes.freeCount = arenaSizeInBytes_ - allocatedBytesCount_;
  stats.statsInBytes.allocatedCount = allocatedBytesCount_;
  stats.statsInBytes.arenaSize = arenaSizeInBytes_;
  stats.statsInBytes.allocatedRatio =
      static_cast<double>(allocatedBytesCount_) / arenaSizeInBytes_;

  stats.statsInBlocks.arenaSize = arenaSizeInBlocks_;
  stats.statsInBlocks.freeCount = arenaSizeInBlocks_ - allocatedBlocksCount_;
  stats.statsInBlocks.allocatedCount = allocatedBlocksCount_;
  stats.statsInBlocks.arenaSize = arenaSizeInBlocks_;
  stats.statsInBlocks.allocatedRatio =
      static_cast<double>(allocatedBlocksCount_) / arenaSizeInBlocks_;

  stats.arena = arena_;
  stats.blockSize = blockSize_;
  stats.allocationsCount = totalNumberOfAllocations_;
  stats.deAllocationsCount = totalNumberOfFree_;

  const double bytesAllocatedOutOfBlocksAllocated = allocatedBytesCount_
      ? (static_cast<double>(allocatedBytesCount_) /
         static_cast<double>(allocatedBlocksCount_ * blockSize_))
      : 1.0;

  stats.internalFragmentationScore = 1.0 - bytesAllocatedOutOfBlocksAllocated;
  stats.externalFragmentationScore = 0;

  return stats;
}

size_t MemoryPool::getAllocatedSizeInBytes(void* ptr) const {
  size_t offset = (static_cast<char*>(ptr) - static_cast<char*>(arena_));
  if (offset < 0 || offset >= arenaSizeInBlocks_ || (offset % blockSize_)) {
    std::stringstream stream;
    stream << "MemoryPool::getAllocatedSizeInBytes(ptr=" << ptr
           << ") pointer is not managed by MemoryPool. offset=" << offset;
    throw std::invalid_argument(stream.str());
  }
  return blockSize_;
}

}; // namespace fl
