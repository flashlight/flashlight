/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/memory/allocator/freelist/FreeList.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "flashlight/common/Histogram.h"
#include "flashlight/common/Logging.h"
#include "flashlight/common/Utils.h"

namespace fl {

// prettyString() will show up to the first kPrettyStringMaxChunks chunks.
constexpr size_t kPrettyStringMaxChunks = 3;
constexpr size_t kHistBucketCountPrettyString = 15;

FreeList::Chunk::Chunk(size_t blockStartIndex, size_t sizeInBlocks)
    : blockStartIndex_(blockStartIndex),
      sizeInBlocks_(sizeInBlocks),
      next_(nullptr),
      prev_(nullptr) {}

std::string FreeList::Chunk::prettyString() const {
  std::stringstream stream;
  stream << "Chunk{blockStartIndex_=" << blockStartIndex_
         << " sizeInBlocks_=" << sizeInBlocks_ << " next_=" << next_
         << " prev_=" << prev_ << "}";
  return stream.str();
}

FreeList::FreeList(
    std::string name,
    void* arena,
    size_t arenaSizeInBytes,
    size_t blockSize)
    : MemoryAllocator(std::move(name)),
      arena_(arena),
      arenaSizeInBytes_(arenaSizeInBytes),
      arenaSizeInBlocks_(arenaSizeInBytes / blockSize),
      blockSize_(blockSize),
      allocatedBytesCount_(0),
      allocatedBlocksCount_(0),
      largestChunkInBlocks_(arenaSizeInBlocks_),
      totalNumberOfChunks_(1),
      totalNumberOfAllocations_(0),
      totalNumberOfFree_(0),
      freeListSize_(1),
      freeList_(new Chunk(0, arenaSizeInBlocks_)) {
  LOG(INFO) << "Created " << prettyString();
}

FreeList::~FreeList() {
  if (pointerToAllocatedChunkMap_.size()) {
    LOG(WARNING) << "~FreeList() there are still " << allocatedBlocksCount_
                 << " blocks in " << pointerToAllocatedChunkMap_.size()
                 << " chunks held by the user";
    for (auto& ptrAndChunk : pointerToAllocatedChunkMap_) {
      delete ptrAndChunk.second;
    }
  }
}

void* FreeList::allocate(size_t size) {
  if (!size) {
    return nullptr;
  }
  const size_t nBlocksNeed = divRoundUp(size, blockSize_);

  Chunk* chunk = findBestFit(nBlocksNeed);
  chopChunkIfNeeded(chunk, nBlocksNeed);
  removeFromFreeList(chunk);
  void* ptr = memoryAddress(arena_, chunk->blockStartIndex_);
  pointerToAllocatedChunkMap_[ptr] = chunk;

  // stats
  chunk->sizeInBytes_ = size;
  allocatedBytesCount_ += size;
  allocatedBlocksCount_ += chunk->sizeInBlocks_;
  --freeListSize_;
  historyOfAllocationRequestInBlocks_.push_back(nBlocksNeed);

  VLOG(2) << "FreeList::allocate(size=" << size << ") return ptr=" << ptr
          << " name=" << getName() << " nBlocksNeed=" << nBlocksNeed;

  ++totalNumberOfAllocations_;

  if (!ptr) {
    LOG(ERROR) << "FreeList::allocate(size_t size) retusn null";
  }
  return ptr;
}

namespace {
void throwGetChunkError(
    void* ptr,
    const std::string& callerNameForErrorMessage,
    const std::string& name) {
  std::stringstream ss;
  ss << callerNameForErrorMessage << "(ptr=" << ptr
     << ") pointer is not managed by FreeList name=" << name;
  throw std::invalid_argument(ss.str());
}
} // namespace

FreeList::Chunk* FreeList::getChunkAndRemoveFromMap(
    void* ptr,
    const std::string& callerNameForErrorMessage) {
  Chunk* chunk = nullptr;
  auto chunkItr = pointerToAllocatedChunkMap_.find(ptr);
  if (chunkItr != pointerToAllocatedChunkMap_.end()) {
    chunk = chunkItr->second;
    pointerToAllocatedChunkMap_.erase(chunkItr);
  } else {
    throwGetChunkError(ptr, callerNameForErrorMessage, getName());
  }
  return chunk;
}

const FreeList::Chunk* FreeList::getChunk(
    void* ptr,
    const std::string& callerNameForErrorMessage) const {
  const auto chunkItr = pointerToAllocatedChunkMap_.find(ptr);
  if (chunkItr == pointerToAllocatedChunkMap_.end()) {
    throwGetChunkError(ptr, callerNameForErrorMessage, getName());
  }
  return chunkItr->second;
}

void FreeList::free(void* ptr) {
  Chunk* chunk = getChunkAndRemoveFromMap(ptr, __PRETTY_FUNCTION__);
  VLOG(2) << "FreeList::free(ptr=" << ptr << ") name=" << getName()
          << " chunk=" << chunk->prettyString();

  // stats
  allocatedBytesCount_ -= chunk->sizeInBytes_;
  allocatedBlocksCount_ -= chunk->sizeInBlocks_;
  ++totalNumberOfFree_;

  addToFreeList(chunk);
  mergeWithNeighbors(chunk);
}

std::string FreeList::prettyString() const {
  const Stats stats = getStats();
  std::stringstream stream;
  stream << "FreeList{name=" << getName() << " stats=" << stats.prettyString()
         << " largestChunkInBlocks_=" << largestChunkInBlocks_
         << " totalNumberOfChunks_=" << totalNumberOfChunks_
         << " numberOfAllocatedChunks=" << pointerToAllocatedChunkMap_.size()
         << " freeListSize_=" << freeListSize_ << " freeList_={";
  int chunkNum = 0;
  for (Chunk *cur = freeList_; cur && (chunkNum < kPrettyStringMaxChunks);
       cur = cur->next_, ++chunkNum) {
    stream << cur->prettyString() << " ";
  }
  if (freeListSize_ > kPrettyStringMaxChunks) {
    stream << "... not showing the next_ "
           << (freeListSize_ - kPrettyStringMaxChunks) << " chunks.";
  }
  {
    std::vector<size_t> cureentAllocationSize(
        pointerToAllocatedChunkMap_.size());
    for (const auto& ptrAndChunk : pointerToAllocatedChunkMap_) {
      cureentAllocationSize.push_back(ptrAndChunk.second->sizeInBytes_);
    }
    HistogramStats<size_t> hist = FixedBucketSizeHistogram<size_t>(
        cureentAllocationSize.begin(),
        cureentAllocationSize.end(),
        kHistBucketCountPrettyString);
    stream << std::endl << "Currently allocated:\n";
    stream << hist.prettyString();
    stream << std::endl;
  }
  stream << "}";

  return stream.str();
}

FreeList::Chunk* FreeList::findBestFit(size_t nBlocksNeed) {
  size_t smallestMatchSize = SIZE_MAX;
  Chunk* bestFit = nullptr;
  largestChunkInBlocks_ = 0;

  for (Chunk* cur = freeList_; cur; cur = cur->next_) {
    if (cur->sizeInBlocks_ >= nBlocksNeed &&
        cur->sizeInBlocks_ <= smallestMatchSize) {
      bestFit = cur;
      smallestMatchSize = cur->sizeInBlocks_;
    }
    largestChunkInBlocks_ = std::max(largestChunkInBlocks_, cur->sizeInBlocks_);
  }

  if (bestFit) {
    VLOG(3) << "FreeList::findBestFit(nBlocksNeed=" << nBlocksNeed << ")"
            << " name=" << getName() << " return " << bestFit->prettyString()
            << ")";
  } else {
    std::stringstream ss;
    ss << "OOM error at FreeList::findBestFit(nBlocksNeed=" << nBlocksNeed
       << ") failed to allocate since largestChunkInBlocks_="
       << largestChunkInBlocks_
       << " which is smaller than requested number of blocks."
       << prettyString();
    LOG(ERROR) << ss.str();
    throw std::invalid_argument(ss.str());
  }

  return bestFit;
}

void FreeList::chopChunkIfNeeded(FreeList::Chunk* chunk, size_t nBlocksNeed) {
  const Chunk origChunk = *chunk;
  if (chunk->sizeInBlocks_ > nBlocksNeed) {
    Chunk* newChunk = new Chunk(
        chunk->blockStartIndex_ + nBlocksNeed,
        chunk->sizeInBlocks_ - nBlocksNeed);
    newChunk->next_ = chunk->next_;
    newChunk->prev_ = chunk;
    if (chunk->next_) {
      chunk->next_->prev_ = newChunk;
    }
    chunk->next_ = newChunk;
    chunk->sizeInBlocks_ = nBlocksNeed;

    // Stats
    ++freeListSize_;
    ++totalNumberOfChunks_;

    VLOG(3) << "FreeList::chopChunkIfNeeded(chunk=" << origChunk.prettyString()
            << " nBlocksNeed=" << nBlocksNeed << ") name=" << getName()
            << " newChunk=" << newChunk->prettyString();
  }
}

void FreeList::addToFreeList(FreeList::Chunk* chunk) {
  if (!freeList_) {
    chunk->prev_ = nullptr;
    chunk->next_ = nullptr;
    freeList_ = chunk;
  } else if (freeList_->blockStartIndex_ > chunk->blockStartIndex_) {
    chunk->prev_ = nullptr;
    chunk->next_ = freeList_;
    freeList_->prev_ = chunk;
    freeList_ = chunk;
  } else {
    // Find the chunk right before the chunk we are adding.
    Chunk* chunkBeforeNewChunk = freeList_;
    while (chunkBeforeNewChunk->next_ &&
           chunkBeforeNewChunk->next_->blockStartIndex_ <
               chunk->blockStartIndex_) {
      chunkBeforeNewChunk = chunkBeforeNewChunk->next_;
    }

    chunk->prev_ = chunkBeforeNewChunk;
    chunk->next_ = chunkBeforeNewChunk->next_;
    if (chunkBeforeNewChunk->next_) {
      chunkBeforeNewChunk->next_->prev_ = chunk;
    }
    chunkBeforeNewChunk->next_ = chunk;
  }

  ++freeListSize_;
}

FreeList::Chunk* FreeList::mergeWithPrevNeighbors(FreeList::Chunk* chunk) {
  Chunk* prevChunk = chunk->prev_;
  if (prevChunk &&
      (prevChunk->blockStartIndex_ + prevChunk->sizeInBlocks_) ==
          chunk->blockStartIndex_) {
    const size_t origSize = chunk->sizeInBlocks_;

    prevChunk->next_ = chunk->next_;
    prevChunk->sizeInBlocks_ += chunk->sizeInBlocks_;

    Chunk* nextChunk = chunk->next_;
    if (nextChunk) {
      nextChunk->prev_ = prevChunk;
    }
    delete chunk;

    // stats
    --freeListSize_;
    --totalNumberOfChunks_;

    VLOG(2) << "FreeList::mergeWithPrevNeighbors(). original size=" << origSize
            << " new size=" << prevChunk->sizeInBlocks_
            << " new chunk details=" << prevChunk->prettyString() << ")";

    return prevChunk;
  }
  return chunk;
}

void FreeList::mergeWithNextNeighbors(FreeList::Chunk* chunk) {
  Chunk* nextChunk = chunk->next_;
  if (nextChunk &&
      (chunk->blockStartIndex_ + chunk->sizeInBlocks_) ==
          nextChunk->blockStartIndex_) {
    const size_t origSize = chunk->sizeInBlocks_;

    chunk->sizeInBlocks_ += nextChunk->sizeInBlocks_;
    chunk->next_ = nextChunk->next_;
    if (nextChunk->next_) {
      nextChunk->next_->prev_ = chunk;
    }
    delete nextChunk;

    // stats
    --freeListSize_;
    --totalNumberOfChunks_;

    VLOG(2) << "FreeList::mergeWithNextNeighbors(). original size=" << origSize
            << " new size=" << chunk->sizeInBlocks_
            << " new chunk details=" << chunk->prettyString() << ")";
  }
}

void FreeList::mergeWithNeighbors(FreeList::Chunk* chunk) {
  chunk = mergeWithPrevNeighbors(chunk);
  mergeWithNextNeighbors(chunk);
}

void FreeList::removeFromFreeList(FreeList::Chunk* chunk) {
  // If chunk is the first in the freeList_.
  if (freeList_ == chunk) {
    freeList_ = chunk->next_;
  }

  if (chunk->prev_) {
    chunk->prev_->next_ = chunk->next_;
  } else {
    freeList_ = chunk->next_;
  }
  if (chunk->next_) {
    chunk->next_->prev_ = chunk->prev_;
  }

  chunk->prev_ = nullptr;
  chunk->next_ = nullptr;
}

MemoryAllocator::Stats FreeList::getStats() const {
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

  const double bytesAllocatedOutOfBlocksAllocated = allocatedBlocksCount_
      ? (static_cast<double>(allocatedBytesCount_) /
         static_cast<double>(allocatedBlocksCount_ * blockSize_))
      : 1.0;

  largestChunkInBlocks_ = 0;
  for (Chunk* cur = freeList_; cur; cur = cur->next_) {
    largestChunkInBlocks_ = std::max(largestChunkInBlocks_, cur->sizeInBlocks_);
  }

  const double largetsChunkOutOfFreeChunks = stats.statsInBlocks.freeCount
      ? (static_cast<double>(largestChunkInBlocks_) /
         static_cast<double>(stats.statsInBlocks.freeCount))
      : 1.0;

  stats.internalFragmentationScore = 1.0 - bytesAllocatedOutOfBlocksAllocated;
  stats.externalFragmentationScore = 1.0 - largetsChunkOutOfFreeChunks;

  return stats;
}

size_t FreeList::getAllocatedSizeInBytes(void* ptr) const {
  const Chunk* chunk = getChunk(ptr, __PRETTY_FUNCTION__);
  return chunk->sizeInBytes_;
}

}; // namespace fl
