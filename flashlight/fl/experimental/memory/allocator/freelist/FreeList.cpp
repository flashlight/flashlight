/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/experimental/memory/allocator/freelist/FreeList.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "flashlight/fl/common/Histogram.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/common/Utils.h"

namespace fl {

// prettyString() will show up to the first kPrettyStringMaxChunks chunks.
constexpr size_t kPrettyStringMaxChunks = 3;

FreeList::Chunk::Chunk(size_t blockStartIndex, size_t sizeInBlocks)
    : blockStartIndex_(blockStartIndex),
      sizeInBlocks_(sizeInBlocks),
      next_(nullptr),
      prev_(nullptr) {}

std::string FreeList::Chunk::prettyString() const {
  std::stringstream ss;
  ss << "Chunk{blockStartIndex_=" << blockStartIndex_
     << " sizeInBlocks_=" << sizeInBlocks_ << " next_=" << next_
     << " prev_=" << prev_ << "}";
  return ss.str();
}

FreeList::FreeList(
    std::string name,
    void* arena,
    size_t arenaSizeInBytes,
    size_t blockSize,
    double allocatedRatioJitThreshold,
    int logLevel)
    : MemoryAllocator(std::move(name), logLevel),
      stats_(arena, arenaSizeInBytes, blockSize),
      largestChunkInBlocks_(stats_.statsInBlocks.arenaSize),
      totalNumberOfChunks_(1),
      freeListSize_(1),
      maxFreeListSize_(1),
      freeList_(new Chunk(0, stats_.statsInBlocks.arenaSize)),
      allocatedRatioJitThreshold_(allocatedRatioJitThreshold),
      isNullAllocator_(arenaSizeInBytes < blockSize) {
  if (getLogLevel() > 1) {
    std::stringstream ss;
    ss << "FreeList::FreeList(name_=" << name_ << " arena=" << arena
       << " arenaSizeInBytes=" << arenaSizeInBytes << " blockSize=" << blockSize
       << ')';
    if (arenaSizeInBytes < blockSize) {
      ss << " arenaSizeInBytes < blockSize --> isNullAllocator_=true";
      isNullAllocator_ = true;
    }

    ss << ' ' << prettyString();
    FL_LOG(fl::INFO) << ss.str();
  }
}

FreeList::~FreeList() {
  if (pointerToAllocatedChunkMap_.size() && getLogLevel() > 0) {
    FL_LOG(fl::WARNING) << "~FreeList() there are still "
                        << stats_.statsInBlocks.arenaSize << " blocks in "
                        << pointerToAllocatedChunkMap_.size()
                        << " chunks held by the user";
    // Free chunks held by the user.
    for (auto& ptrAndChunk : pointerToAllocatedChunkMap_) {
      delete ptrAndChunk.second;
    }
  }
  // Free chunks in the free list.
  for (Chunk* chunk = freeList_; chunk;) {
    Chunk* toDelete = chunk;
    chunk = chunk->next_;
    delete toDelete;
  }
}

bool FreeList::jitTreeExceedsMemoryPressure(size_t bytes) {
  if (bytes == 0 || isNullAllocator_) {
    return false;
  }
  const size_t nBlocksNeed = divRoundUp(bytes, stats_.blockSize);
  if (nBlocksNeed > largestChunkInBlocks_) {
    FL_LOG(fl::INFO) << "FreeList::jitTreeExceedsMemoryPressure(bytes=" << bytes
                     << ") true nBlocksNeed=" << nBlocksNeed
                     << " largestChunkInBlocks_=" << largestChunkInBlocks_;
    return true;
  }

  const double allocatedRatio =
      static_cast<double>(stats_.statsInBlocks.allocatedCount + nBlocksNeed) /
      static_cast<double>(stats_.statsInBlocks.arenaSize);

  if (allocatedRatio > allocatedRatioJitThreshold_) {
    FL_LOG(fl::INFO) << "FreeList::jitTreeExceedsMemoryPressure(bytes=" << bytes
                     << ") true allocatedRatio=" << allocatedRatio
                     << " allocatedRatioJitThreshold_="
                     << allocatedRatioJitThreshold_;
  }

  return allocatedRatio > allocatedRatioJitThreshold_;
}

void* FreeList::allocate(size_t size) {
  if (!size && isNullAllocator_) {
    return nullptr;
  }
  const size_t nBlocksNeed = divRoundUp(size, stats_.blockSize);

  Chunk* chunk = findBestFit(nBlocksNeed);
  if (!chunk || freeListSize_ == 0) {
    if (getLogLevel() > 1) {
      stats_.incrementOomEventCount();
      std::stringstream ss;
      ss << "OOM error at FreeList::allocate(size="
         << prettyStringMemorySize(size) << ") name=" << getName()
         << " freeListSize_=" << freeListSize_
         << " failed to allocate since largestChunkInBlocks_="
         << largestChunkInBlocks_ << " but nBlocksNeed=" << nBlocksNeed;
      FL_LOG(fl::ERROR) << ss.str();
      throw std::invalid_argument(ss.str());
    }
    return nullptr;
  }
  chopChunkIfNeeded(chunk, nBlocksNeed);
  removeFromFreeList(chunk);
  void* ptr = memoryAddress(stats_.arena, chunk->blockStartIndex_);
  pointerToAllocatedChunkMap_[ptr] = chunk;

  chunk->sizeInBytes_ = size;
  stats_.allocate(chunk->sizeInBytes_, chunk->sizeInBlocks_);
  stats_.addPerformanceCost(freeListSize_);
  --freeListSize_;
  // historyOfAllocationRequestInBlocks_.push_back(nBlocksNeed);

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
  if (getLogLevel() > 1) {
    FL_VLOG(2) << "FreeList::free(ptr=" << ptr << ") name=" << getName()
               << " chunk=" << chunk->prettyString();
  }

  // stats
  stats_.free(chunk->sizeInBytes_, chunk->sizeInBlocks_);
  stats_.addPerformanceCost(freeListSize_);

  addToFreeList(chunk);
  mergeWithNeighbors(chunk);
}

namespace {
constexpr size_t kBlockShowSizeThreshold = 10;
constexpr double kBlockSizeLogBase = 1.2;
const double kBlockSizeRatio = (1.0 / (std::log(kBlockSizeLogBase) * 2.0));

std::string formatBlock(bool isFree, size_t size) {
  const char symbol = (isFree) ? ' ' : 'x';

  std::stringstream ss;
  if (size <= kBlockShowSizeThreshold) {
    for (int i = 0; i < size; ++i) {
      ss << symbol;
    }
  } else {
    const int len = static_cast<int>(std::log(size) * kBlockSizeRatio);
    for (int i = 0; i < len; ++i) {
      ss << symbol;
    }
    ss << '(' << (isFree ? "free" : "used") << '=' << prettyStringCount(size)
       << ')';
    for (int i = 0; i < len; ++i) {
      ss << symbol;
    }
  }

  return ss.str();
};

} // namespace

std::string FreeList::BlockMapFormatter::prettyString() const {
  if (!chunk) {
    return "BlockMapFormatter::prettyString() null chunk error";
  }
  return formatBlock(isFree, chunk->sizeInBlocks_);
}

std::string FreeList::blockMapPrettyString() const {
  std::vector<BlockMapFormatter> chunksFormatter;
  for (const Chunk* cur = freeList_; cur; cur = cur->next_) {
    chunksFormatter.push_back({true, cur});
  }
  for (const auto& ptrAndChunk : pointerToAllocatedChunkMap_) {
    chunksFormatter.push_back({false, ptrAndChunk.second});
  }
  std::sort(
      chunksFormatter.begin(),
      chunksFormatter.end(),
      [](const BlockMapFormatter& lhs, const BlockMapFormatter& rhs) {
        return lhs.chunk->blockStartIndex_ < rhs.chunk->blockStartIndex_;
      });

  std::vector<std::pair<bool, size_t>> chunksConsolidated;
  chunksConsolidated.push_back({chunksFormatter.at(0).isFree,
                                chunksFormatter.at(0).chunk->sizeInBlocks_});
  for (size_t i = 1; i < chunksFormatter.size(); ++i) {
    BlockMapFormatter& cur = chunksFormatter[i];
    if (((cur.isFree) ? 1 : 0) == ((chunksConsolidated.back().first) ? 1 : 0)) {
      chunksConsolidated.back().second += cur.chunk->sizeInBlocks_;
    } else {
      chunksConsolidated.push_back({cur.isFree, cur.chunk->sizeInBlocks_});
    }
  }

  std::stringstream ss;
  ss << "Block-Map (Key: free=' ' used='x'):" << std::endl;
  std::string ret = ss.str();
  for (const auto& freeAndSize : chunksConsolidated) {
    ret = ret + formatBlock(freeAndSize.first, freeAndSize.second);
  }
  return ret;
}

std::string FreeList::prettyString() const {
  const Stats stats = getStats();
  std::stringstream ss;
  ss << "FreeList{name=" << getName() << " stats=" << stats_.prettyString()
     << " largestChunkInBlocks_=" << largestChunkInBlocks_
     << " totalNumberOfChunks_=" << totalNumberOfChunks_
     << " numberOfAllocatedChunks=" << pointerToAllocatedChunkMap_.size()
     << " maxFreeListSize_=" << maxFreeListSize_
     << " freeListSize_=" << freeListSize_ << " freeList_={";
  int chunkNum = 0;
  for (Chunk* cur = freeList_; cur && (chunkNum < kPrettyStringMaxChunks);
       cur = cur->next_, ++chunkNum) {
    ss << cur->prettyString() << " ";
  }
  if (freeListSize_ > kPrettyStringMaxChunks) {
    ss << "... not showing the next_ "
       << (freeListSize_ - kPrettyStringMaxChunks) << " chunks.";
  }
  // Histogram and memory map.
  {
    std::vector<size_t> cureentAllocationSize(
        pointerToAllocatedChunkMap_.size());
    for (const auto& ptrAndChunk : pointerToAllocatedChunkMap_) {
      cureentAllocationSize.push_back(ptrAndChunk.second->sizeInBytes_);
    }
    HistogramStats<size_t> hist = FixedBucketSizeHistogram<size_t>(
        cureentAllocationSize.begin(),
        cureentAllocationSize.end(),
        kHistogramBucketCountPrettyString);
    ss << std::endl
       << "FreeList currently allocated histogram:" << std::endl
       << hist.prettyString() << std::endl
       << "FreeList currently allocated memory map:" << std::endl
       << blockMapPrettyString() << std::endl;
  }
  ss << "}";

  return ss.str();
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

  if (getLogLevel() > 1) {
    FL_VLOG(3) << "FreeList::findBestFit(nBlocksNeed=" << nBlocksNeed << ")"
               << " name=" << getName() << " return "
               << (bestFit ? bestFit->prettyString() : "null") << ')';
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
    maxFreeListSize_ = std::max(maxFreeListSize_, freeListSize_);
    ++totalNumberOfChunks_;

    if (getLogLevel() > 1) {
      FL_VLOG(3) << "FreeList::chopChunkIfNeeded(chunk="
                 << origChunk.prettyString() << " nBlocksNeed=" << nBlocksNeed
                 << ") name=" << getName()
                 << " newChunk=" << newChunk->prettyString();
    }
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
  maxFreeListSize_ = std::max(maxFreeListSize_, freeListSize_);
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
    if (freeListSize_ == 0 || totalNumberOfChunks_ == 0) {
      std::stringstream ss;
      ss << "FreeList::mergeWithPrevNeighbors() freeListSize_=" << freeListSize_
         << " totalNumberOfChunks_=" << totalNumberOfChunks_
         << " but both should be > 1.";
      FL_LOG(fl::ERROR) << ss.str();
      throw std::runtime_error(ss.str());
    }
    --freeListSize_;
    --totalNumberOfChunks_;

    if (getLogLevel() > 1) {
      FL_VLOG(2) << "FreeList::mergeWithPrevNeighbors(). original size="
                 << origSize << " new size=" << prevChunk->sizeInBlocks_
                 << " new chunk details=" << prevChunk->prettyString() << ")";
    }

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
    if (freeListSize_ == 0 || totalNumberOfChunks_ == 0) {
      std::stringstream ss;
      ss << "FreeList::mergeWithNextNeighbors() freeListSize_=" << freeListSize_
         << " totalNumberOfChunks_=" << totalNumberOfChunks_
         << " but both should be > 1.";
      FL_LOG(fl::ERROR) << ss.str();
      throw std::runtime_error(ss.str());
    }
    --freeListSize_;
    --totalNumberOfChunks_;

    if (getLogLevel() > 1) {
      FL_VLOG(2) << "FreeList::mergeWithNextNeighbors(). original size="
                 << origSize << " new size=" << chunk->sizeInBlocks_
                 << " new chunk details=" << chunk->prettyString() << ")";
    }
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
  // Calculate ExternalFragmentationScore
  largestChunkInBlocks_ = 0;
  for (Chunk* cur = freeList_; cur; cur = cur->next_) {
    largestChunkInBlocks_ = std::max(largestChunkInBlocks_, cur->sizeInBlocks_);
  }
  const double largetsChunkOutOfFreeChunks =
      (stats_.statsInBlocks.freeCount > 0)
      ? (static_cast<double>(largestChunkInBlocks_) /
         static_cast<double>(stats_.statsInBlocks.freeCount))
      : 1.0;
  stats_.setExternalFragmentationScore(1.0 - largetsChunkOutOfFreeChunks);

  return stats_;
}

size_t FreeList::getAllocatedSizeInBytes(void* ptr) const {
  const Chunk* chunk = getChunk(ptr, __PRETTY_FUNCTION__);
  return chunk->sizeInBytes_;
}

}; // namespace fl
