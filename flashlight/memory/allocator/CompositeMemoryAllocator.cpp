/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/memory/allocator/CompositeMemoryAllocator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "flashlight/common/Logging.h"

namespace fl {
constexpr size_t numberOfAllocationsBetweenPrettyString = 1000;

std::string CompositeMemoryAllocator::AllocatorAndCriteria::prettyString()
    const {
  std::stringstream stream;
  stream << "CompositeMemoryAllocator{"
         << " maxAllocationSize=" << maxAllocationSize
         << " allocator=" << allocator->prettyString();
  return stream.str();
}

CompositeMemoryAllocator::CompositeMemoryAllocator()
    : totalNumberOfAllocations_(0),
      arenaSizeInBlocks_(0),
      arenaSizeInBytes_(0) {}

void CompositeMemoryAllocator::add(AllocatorAndCriteria allocatorAndCriteria) {
  const MemoryAllocator::Stats stats =
      allocatorAndCriteria.allocator->getStats();
  arenaSizeInBlocks_ += stats.statsInBlocks.arenaSize;
  arenaSizeInBytes_ += stats.statsInBytes.arenaSize;
  allocatorsAndCriterias_.push_back(std::move(allocatorAndCriteria));
  std::sort(allocatorsAndCriterias_.begin(), allocatorsAndCriterias_.end());
}

void* CompositeMemoryAllocator::allocate(size_t size) {
  VLOG(2) << "CompositeMemoryAllocator::allocate(size=" << size << ")";
  if (size == 0) {
    return nullptr;
  }
  void* ptr = nullptr;
  std::vector<std::exception> exceptions;
  for (size_t i = 0; i < allocatorsAndCriterias_.size() && !ptr; ++i) {
    if (allocatorsAndCriterias_[i].maxAllocationSize >= size) {
      try {
        ptr = allocatorsAndCriterias_[i].allocator->allocate(size);
        ptrToAllocatorsAndCriteriasIndex_[ptr] = i;
        VLOG(2) << "CompositeMemoryAllocator::allocate(size=" << size
                << ") i=" << i << " ptr=" << ptr;
      } catch (std::exception ex) {
        // Catch the OOM exception because maybe the next allocator can satisfy
        // the request.
        VLOG(1) << "CompositeMemoryAllocator::allocate(size=" << size
                << ") i=" << i << " exception=" << ex.what();
        exceptions.push_back(ex);
      }
    }
  }

  if (!ptr) {
    std::stringstream stream;
    stream << "CompositeMemoryAllocator::allocateImpl(size=" << size
           << ") failed to allocate with errors={";
    for (const std::exception& ex : exceptions) {
      stream << '{' << ex.what() << "},";
    }
    stream << "}";
    throw std::invalid_argument(stream.str());
  }

  ++totalNumberOfAllocations_;
  if (!(totalNumberOfAllocations_ % numberOfAllocationsBetweenPrettyString)) {
    LOG(INFO) << prettyString();
  }

  return ptr;
}

void CompositeMemoryAllocator::free(void* ptr) {
  auto itr = ptrToAllocatorsAndCriteriasIndex_.find(ptr);
  if (itr != ptrToAllocatorsAndCriteriasIndex_.end()) {
    allocatorsAndCriterias_[itr->second].allocator->free(ptr);
  } else {
    std::stringstream ss;
    ss << "CompositeMemoryAllocator::free(ptr=" << ptr
       << ") pointer is not managed allocator.";
    throw std::invalid_argument(ss.str());
  }
}

std::string CompositeMemoryAllocator::prettyString() const {
  std::stringstream stream;
  stream << "CompositeMemoryAllocator{totalNumberOfAllocations_="
         << totalNumberOfAllocations_
         << " numberOfAllocators=" << allocatorsAndCriterias_.size()
         << " allcatorsAndCriterias={" << std::endl;
  for (const AllocatorAndCriteria& allocatorAndCriteria :
       allocatorsAndCriterias_) {
    stream << allocatorAndCriteria.prettyString() << std::endl;
  }

  stream << "}";
  return stream.str();
}

size_t CompositeMemoryAllocator::getAllocatedSizeInBytes(void* ptr) const {
  auto itr = ptrToAllocatorsAndCriteriasIndex_.find(ptr);
  if (itr != ptrToAllocatorsAndCriteriasIndex_.end()) {
    return allocatorsAndCriterias_[itr->second]
        .allocator->getAllocatedSizeInBytes(ptr);
  } else {
    std::stringstream ss;
    ss << "CompositeMemoryAllocator::getAllocatedSizeInBytes(ptr=" << ptr
       << ") pointer is not managed allocator.";
    throw std::invalid_argument(ss.str());
  }
}

MemoryAllocator::Stats CompositeMemoryAllocator::getStats() const {
  Stats stats = {};

  // Arena is set to the minimum arena of all allocators;
  stats.arena =
      reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max());

  stats.statsInBlocks.arenaSize = arenaSizeInBlocks_;
  stats.statsInBytes.arenaSize = arenaSizeInBytes_;
  stats.allocationsCount = totalNumberOfAllocations_;

  double blockSize = 0.0;
  for (const AllocatorAndCriteria& allocatorAndCriteria :
       allocatorsAndCriterias_) {
    const MemoryAllocator::Stats curStats =
        allocatorAndCriteria.allocator->getStats();
    // Weight is the current allocator arena size relative to sum of all
    // arenas.
    const double weight = static_cast<double>(curStats.statsInBytes.arenaSize) /
        arenaSizeInBytes_;

    stats.arena = std::min(stats.arena, curStats.arena);
    blockSize += static_cast<double>(curStats.blockSize) * weight;
    stats.deAllocationsCount += curStats.deAllocationsCount;
    stats.externalFragmentationScore +=
        curStats.externalFragmentationScore * weight;
    stats.internalFragmentationScore +=
        curStats.internalFragmentationScore * weight;
    LOG(INFO) << "stats.externalFragmentationScore="
              << stats.externalFragmentationScore << " weight=" << weight
              << " curStats.externalFragmentationScore="
              << curStats.externalFragmentationScore
              << " (curStats.externalFragmentationScore * weight)="
              << (curStats.externalFragmentationScore * weight);

    stats.statsInBytes.allocatedCount += curStats.statsInBytes.allocatedCount;
    stats.statsInBlocks.allocatedCount += curStats.statsInBlocks.allocatedCount;
  }

  // blockSize is set to weighted average of all allocator. The weight
  // is determined by arena size.
  stats.blockSize = blockSize;

  stats.statsInBytes.freeCount =
      stats.statsInBytes.arenaSize - stats.statsInBytes.allocatedCount;
  stats.statsInBytes.allocatedRatio =
      static_cast<double>(stats.statsInBytes.allocatedCount) /
      stats.statsInBytes.arenaSize;

  stats.statsInBlocks.freeCount =
      stats.statsInBlocks.arenaSize - stats.statsInBlocks.allocatedCount;
  stats.statsInBlocks.allocatedRatio =
      static_cast<double>(stats.statsInBlocks.allocatedCount) /
      stats.statsInBlocks.arenaSize;

  return stats;
}
}; // namespace fl
