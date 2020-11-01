/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/experimental/memory/allocator/CompositeMemoryAllocator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "flashlight/fl/common/Histogram.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/common/Utils.h"

namespace fl {

std::string CompositeMemoryAllocator::AllocatorAndCriteria::prettyString()
    const {
  std::stringstream ss;
  ss << "AllocatorAndCriteria{" << std::endl
     << "maxAllocationSize=" << prettyStringMemorySize(maxAllocationSize)
     << std::endl
     << " allocator=" << allocator->prettyString() << std::endl
     << '}';
  return ss.str();
}

CompositeMemoryAllocator::CompositeMemoryAllocator(std::string name)
    : MemoryAllocator(name),
      totalNumberOfAllocations_(0),
      arenaSizeInBlocks_(0),
      arenaSizeInBytes_(0),
      failToAllocate_(false),
      oomEventCount_(0) {}

void CompositeMemoryAllocator::add(AllocatorAndCriteria allocatorAndCriteria) {
  const MemoryAllocator::Stats stats =
      allocatorAndCriteria.allocator->getStats();
  arenaSizeInBlocks_ += stats.statsInBlocks.arenaSize;
  arenaSizeInBytes_ += stats.statsInBytes.arenaSize;
  allocatorsAndCriterias_.push_back(std::move(allocatorAndCriteria));
  std::sort(allocatorsAndCriterias_.begin(), allocatorsAndCriterias_.end());
}

bool CompositeMemoryAllocator::jitTreeExceedsMemoryPressure(size_t bytes) {
  if (bytes == 0) {
    return false;
  }
  for (size_t i = 0; i < allocatorsAndCriterias_.size(); ++i) {
    if (allocatorsAndCriterias_[i].maxAllocationSize >= bytes) {
      return allocatorsAndCriterias_[i].allocator->jitTreeExceedsMemoryPressure(
          bytes);
    }
  }
  return false;
}

void* CompositeMemoryAllocator::allocate(size_t size) {
  if (size == 0) {
    return nullptr;
  }
  void* ptr = nullptr;
  std::vector<std::string> errors;
  for (size_t i = 0; i < allocatorsAndCriterias_.size(); ++i) {
    // Try allocate if size is below the allocators maxAllocationSize.
    if (allocatorsAndCriterias_[i].maxAllocationSize >= size) {
      try {
        ptr = allocatorsAndCriterias_[i].allocator->allocate(size);
      } catch (std::exception& ex) {
        // Catch the OOM exception because maybe the next allocator can satisfy
        // the request.

        // errors.push_back(ex.what());
      }

      if (ptr) {
        ptrToAllocation_[ptr] = {size, i};
        break;
      }
    }
  }

  // When getLogLevel()==0 only log when ptr==0
  std::string oomErrorDescription;
  if (!ptr || (!errors.empty())) {
    std::stringstream ss;
    failToAllocate_ = true;

    Stats stats = getStats();
    const size_t totalFreeMem =
        stats.statsInBytes.arenaSize - stats.statsInBytes.allocatedCount;

    ss << "CompositeMemoryAllocator::allocateImpl(size="
       << prettyStringMemorySize(size) << ')'
       << " totalFreeMem=" << prettyStringMemorySize(totalFreeMem)
       << " (totalFreeMem - size)="
       << prettyStringMemorySize(totalFreeMem - size)
       << " (size / totalFreeMem)="
       << (static_cast<double>(size) / totalFreeMem) << " experienced a";
    if (ptr) {
      ss << " partial";
    } else {
      ss << " !full!";
    }
    ss << " allocation error={";
    for (const std::string& error : errors) {
      ss << '{' << error << "},";
    }

    ss << '}' << std::endl;
    ss << "Dumping memory state at time of OOM:" << std::endl << prettyString();
    oomErrorDescription = ss.str();

    if (ptr) {
      if (getLogLevel() > 1) {
        FL_LOG(fl::WARNING) << oomErrorDescription;
      }
    } else {
      if (getLogLevel() > 0) {
        FL_LOG(fl::ERROR) << oomErrorDescription;
      }
      if (!oomEventCount_) {
        firstOomStats_.stats = getStats();
        firstOomStats_.allocationSize = size;
        firstOomStats_.description = oomErrorDescription;
      }
      ++oomEventCount_;
      throw std::invalid_argument(oomErrorDescription);
    }
  }

  ++totalNumberOfAllocations_;
  return ptr;
}

void CompositeMemoryAllocator::free(void* ptr) {
  auto itr = ptrToAllocation_.find(ptr);
  if (itr != ptrToAllocation_.end()) {
    allocatorsAndCriterias_[itr->second.allocatorsAndCriteriasIndex]
        .allocator->free(ptr);
  } else {
    std::stringstream ss;
    ss << "CompositeMemoryAllocator::free(ptr=" << ptr
       << ") pointer is not managed allocator.";
    throw std::invalid_argument(ss.str());
  }
}

CompositeMemoryAllocator::FirstOomStats::FirstOomStats() : allocationSize(0) {}

std::string CompositeMemoryAllocator::FirstOomStats::prettyString() const {
  std::stringstream ss;
  const size_t freeMem =
      stats.statsInBytes.arenaSize - stats.statsInBytes.allocatedCount;
  ss << "allocationSize=" << prettyStringMemorySize(allocationSize)
     << " freeMem=" << prettyStringMemorySize(freeMem)
     << " (free - allocationSize)="
     << prettyStringMemorySize(freeMem - allocationSize)
     << " (allocationSize / free)="
     << (static_cast<double>(allocationSize) / freeMem) << " description={"
     << description << '}';
  return ss.str();
}

std::string CompositeMemoryAllocator::prettyString() const {
  const MemoryAllocator::Stats stats = getStats();
  std::stringstream ss;
  ss << "CompositeMemoryAllocator{stats=" << stats.prettyString() << std::endl;

  if (oomEventCount_) {
    ss << "First OOM event stats={" << firstOomStats_.prettyString() << '}'
       << std::endl;
  }

  if (ptrToAllocation_.empty()) {
    ss << std::endl
       << "CompositeMemoryAllocator currently has no allocations." << std::endl;
  } else {
    ss << std::endl
       << "CompositeMemoryAllocator currenly allocated higtogram:" << std::endl;

    std::vector<size_t> currentAllocationSize(ptrToAllocation_.size());
    for (const auto& ptrAndAllocation : ptrToAllocation_) {
      currentAllocationSize.push_back(ptrAndAllocation.second.size);
    }
    HistogramStats<size_t> hist = FixedBucketSizeHistogram<size_t>(
        currentAllocationSize.begin(),
        currentAllocationSize.end(),
        kHistogramBucketCountPrettyString);
    ss << hist.prettyString() << std::endl;

    // Show double high resolution when bucket[0] have > 100 items.
    if (!hist.buckets.empty() && currentAllocationSize.size() > 100) {
      const HistogramBucket<size_t>& largestCountBucket = hist.buckets[0];
      if (largestCountBucket.count > 100) {
        HistogramStats<size_t> hiResHist = FixedBucketSizeHistogram<size_t>(
            currentAllocationSize.begin(),
            currentAllocationSize.end(),
            kHistogramBucketCountPrettyString,
            largestCountBucket.startInclusive,
            largestCountBucket.endExclusive);

        ss << std::endl
           << "Currently allocated hi-resolution:" << std::endl
           << hiResHist.prettyString() << std::endl;

        // Show double high resolution when bucket[0] have > 100 items.
        if (!hiResHist.buckets.empty()) {
          const HistogramBucket<size_t>& doubleLargestCountBucket =
              hiResHist.buckets[0];
          if (doubleLargestCountBucket.count > 100) {
            HistogramStats<size_t> doubleHiResHist =
                FixedBucketSizeHistogram<size_t>(
                    currentAllocationSize.begin(),
                    currentAllocationSize.end(),
                    kHistogramBucketCountPrettyString,
                    doubleLargestCountBucket.startInclusive,
                    doubleLargestCountBucket.endExclusive);

            ss << std::endl
               << "Currently allocated double hi-resolution:" << std::endl
               << doubleHiResHist.prettyString() << std::endl;
          }
        }
      }
    }
  }

  ss << " allocatorsAndCriterias_.size()=" << allocatorsAndCriterias_.size()
     << " allcatorsAndCriterias={" << std::endl;
  for (const AllocatorAndCriteria& allocatorAndCriteria :
       allocatorsAndCriterias_) {
    ss << std::endl << allocatorAndCriteria.prettyString();
  }
  ss << "}}";
  return ss.str();
}

size_t CompositeMemoryAllocator::getAllocatedSizeInBytes(void* ptr) const {
  auto itr = ptrToAllocation_.find(ptr);
  if (itr != ptrToAllocation_.end()) {
    return itr->second.size;
  } else {
    std::stringstream ss;
    ss << "CompositeMemoryAllocator::getAllocatedSizeInBytes(ptr=" << ptr
       << ") pointer is not managed allocator.";
    throw std::invalid_argument(ss.str());
  }
}

void CompositeMemoryAllocator::setLogLevel(int logLevel) {
  for (const AllocatorAndCriteria& allocatorAndCriteria :
       allocatorsAndCriterias_) {
    allocatorAndCriteria.allocator->setLogLevel(logLevel);
  };
  MemoryAllocator::setLogLevel(logLevel);
}

MemoryAllocator::Stats CompositeMemoryAllocator::getStats() const {
  Stats stats;
  stats.failToAllocate = failToAllocate_;

  // Arena is set to the minimum arena of all allocators;
  stats.arena =
      reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max());

  stats.statsInBlocks.arenaSize = arenaSizeInBlocks_;
  stats.statsInBytes.arenaSize = arenaSizeInBytes_;
  stats.allocationsCount = totalNumberOfAllocations_;
  stats.oomEventCount += oomEventCount_;

  // External fragmentation is taken from the last (the catch all) arena.
  const AllocatorAndCriteria& lastAllocatorAndCriteria =
      allocatorsAndCriterias_.back();
  const Stats lastAllocatorStats =
      lastAllocatorAndCriteria.allocator->getStats();
  stats.externalFragmentationScore +=
      lastAllocatorStats.externalFragmentationScore;
  stats.maxExternalFragmentationScore +=
      lastAllocatorStats.maxExternalFragmentationScore;

  double blockSize = 0.0;
  for (const AllocatorAndCriteria& allocatorAndCriteria :
       allocatorsAndCriterias_) {
    MemoryAllocator::Stats curStats =
        allocatorAndCriteria.allocator->getStats();
    // Weight is the current allocator arena size relative to sum of all
    // arenas.
    const double weight = static_cast<double>(curStats.statsInBytes.arenaSize) /
        arenaSizeInBytes_;

    stats.arena = std::min(stats.arena, curStats.arena);
    blockSize += static_cast<double>(curStats.blockSize) * weight;
    stats.deAllocationsCount += curStats.deAllocationsCount;

    stats.internalFragmentationScore +=
        curStats.internalFragmentationScore * weight;
    stats.maxInternalFragmentationScore +=
        curStats.maxInternalFragmentationScore * weight;

    stats.statsInBytes.allocatedCount += curStats.statsInBytes.allocatedCount;
    stats.statsInBytes.maxAllocatedCount +=
        curStats.statsInBytes.maxAllocatedCount;

    stats.statsInBlocks.allocatedCount += curStats.statsInBlocks.allocatedCount;
    stats.statsInBlocks.maxAllocatedCount +=
        curStats.statsInBlocks.maxAllocatedCount;

    stats.performanceCost += curStats.performanceCost;

    stats.subArenaStats.push_back(std::move(curStats));
  }

  // blockSize is set to weighted average of all allocator. The weight
  // is determined by arena size.
  stats.blockSize = blockSize;

  stats.statsInBytes.freeCount =
      stats.statsInBytes.arenaSize - stats.statsInBytes.allocatedCount;

  stats.statsInBlocks.freeCount =
      stats.statsInBlocks.arenaSize - stats.statsInBlocks.allocatedCount;

  return stats;
}
}; // namespace fl
