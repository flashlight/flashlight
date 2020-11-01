/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "flashlight/fl/experimental/memory/allocator/memorypool/MemoryPool.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "flashlight/fl/common/Histogram.h"
#include "flashlight/fl/common/Logging.h"

namespace fl {

MemoryPool::MemoryPool(
    std::string name,
    void* arena,
    size_t arenaSizeInBytes,
    size_t blockSize,
    double allocatedRatioJitThreshold,
    int logLevel)
    : MemoryAllocator(std::move(name), logLevel),
      stats_(arena, arenaSizeInBytes, blockSize),
      freeListSize_(stats_.statsInBlocks.arenaSize),
      freeList_(-1),
      blocks_(stats_.statsInBlocks.arenaSize),
      allocatedRatioJitThreshold_(allocatedRatioJitThreshold),
      isNullAllocator_(stats_.statsInBlocks.arenaSize < 1) {
  std::stringstream ss;
  ss << "MemoryPool::MemoryPool(name_=" << name_ << " arena=" << arena
     << " arenaSizeInBytes_=" << arenaSizeInBytes << " blockSize=" << blockSize
     << ')';

  if (stats_.statsInBlocks.arenaSize < 1) {
    ss << " --> isNullAllocator_=true";
  } else {
    // Each block point to the next.
    for (size_t i = 0; i < (stats_.statsInBlocks.arenaSize - 1); ++i) {
      blocks_[i].indexOfNext = i + 1;
    }
    // last block points to -1.
    blocks_[stats_.statsInBlocks.arenaSize - 1].indexOfNext = -1;

    // point to block index 0
    freeList_ = 0;
  }

  if (getLogLevel() > 1) {
    ss << ' ' << prettyString();
    FL_LOG(fl::INFO) << ss.str();
  }
}

MemoryPool::~MemoryPool() {
  if (stats_.statsInBlocks.allocatedCount && getLogLevel() > 0) {
    FL_LOG(fl::WARNING) << "~MemoryPool() there are still "
                        << stats_.statsInBlocks.allocatedCount
                        << " blocks_ held by the user";
  }
}

void* MemoryPool::toPtr(size_t index) const {
  if (index < 0) {
    return nullptr;
  }
  return static_cast<char*>(stats_.arena) + index * stats_.blockSize;
}

void* MemoryPool::allocate(size_t size) {
  if (size == 0 || isNullAllocator_) {
    return nullptr;
  }
  if (size > stats_.blockSize || freeList_ < 0 || freeListSize_ == 0) {
    std::stringstream ss;
    ss << "MemoryPool::allocate(size=" << size << ") name=" << getName()
       << " freeListSize_=" << freeListSize_;

    if (size > stats_.blockSize) {
      ss << "allocated size must be <= blockSize=" << stats_.blockSize;
      FL_LOG(fl::ERROR) << ss.str();
      // throw std::invalid_argument(ss.str());
    }

    if (freeList_ < 0 || freeListSize_ == 0) {
      if (getLogLevel() > 1) {
        ss << " OOM (out of memory) error. freeList_=" << freeList_;
        FL_LOG(fl::ERROR) << ss.str();
        // throw std::invalid_argument(ss.str());
      }
      stats_.incrementOomEventCount();
    }
    return nullptr;
  }

  long blockIndex = freeList_;
  Block& block = blocks_[blockIndex];
  block.allocatedSize = size;
  freeList_ = block.indexOfNext;
  block.indexOfNext = -1;

  // stats
  stats_.allocate(size, 1);
  stats_.addPerformanceCost(1);
  --freeListSize_;

  return toPtr(blockIndex);
}

void MemoryPool::free(void* ptr) {
  const size_t offset =
      (static_cast<char*>(ptr) - static_cast<char*>(stats_.arena));
  const size_t blockIndex = offset / stats_.blockSize;
  if (offset < 0 || blockIndex >= stats_.statsInBlocks.arenaSize ||
      (offset % stats_.blockSize)) {
    std::stringstream stream;
    stream << "MemoryPool::free(ptr=" << ptr
           << ") pointer is not managed by MemoryPool. offset=" << offset << " "
           << prettyString();
    throw std::invalid_argument(stream.str());
  }
  Block& block = blocks_[blockIndex];

  stats_.free(block.allocatedSize, 1);
  stats_.addPerformanceCost(1);
  ++freeListSize_;

  block.indexOfNext = freeList_;
  block.allocatedSize = 0;
  freeList_ = blockIndex;
}

bool MemoryPool::jitTreeExceedsMemoryPressure(size_t bytes) {
  if (bytes == 0 || isNullAllocator_ || bytes > stats_.blockSize) {
    return false;
  }
  const double allocatedRatio =
      static_cast<double>(stats_.statsInBlocks.allocatedCount + 1) /
      static_cast<double>(stats_.statsInBlocks.arenaSize);

  if (allocatedRatio > allocatedRatioJitThreshold_) {
    FL_LOG(fl::INFO) << "MemoryPool::jitTreeExceedsMemoryPressure(bytes="
                     << bytes << ") true allocatedRatio=" << allocatedRatio
                     << " allocatedRatioJitThreshold_="
                     << allocatedRatioJitThreshold_;
  }

  return allocatedRatio > allocatedRatioJitThreshold_;
}

std::string MemoryPool::prettyString() const {
  std::stringstream stream;
  stream << "MemoryPool{name=" << getName()
         << " stats_=" << stats_.prettyString();
  stream << " freeListSize_=" << freeListSize_ << " freeList_=" << freeList_;
  {
    HistogramStats<size_t> hist = FixedBucketSizeHistogram<size_t>(
        currentlyAlloctedInBytes_.begin(),
        currentlyAlloctedInBytes_.end(),
        kHistogramBucketCountPrettyString);
    stream << std::endl << "Currently allocated:" << std::endl;
    stream << hist.prettyString();
    stream << std::endl;
  }
  stream << "}";

  return stream.str();
}

MemoryAllocator::Stats MemoryPool::getStats() const {
  return stats_;
}

size_t MemoryPool::getAllocatedSizeInBytes(void* ptr) const {
  size_t offset = (static_cast<char*>(ptr) - static_cast<char*>(stats_.arena));
  if (offset < 0 || offset >= stats_.statsInBlocks.arenaSize ||
      (offset % stats_.blockSize)) {
    std::stringstream stream;
    stream << "MemoryPool::getAllocatedSizeInBytes(ptr=" << ptr
           << ") pointer is not managed by MemoryPool. offset=" << offset;
    throw std::invalid_argument(stream.str());
  }
  const size_t blockIndex = offset / stats_.blockSize;
  return blocks_[blockIndex].allocatedSize;
}

}; // namespace fl
