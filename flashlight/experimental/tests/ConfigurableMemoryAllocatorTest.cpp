/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/common/Logging.h"
#include "flashlight/experimental/memory/allocator/ConfigurableMemoryAllocator.h"

using namespace fl;

namespace {

constexpr size_t alignmentNumberOfBits = 5;

bool isAligned(void* arena, void* ptr) {
  if (arena == ptr) {
    return true;
  }
  size_t diff =
      reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(arena);
  return (diff >> alignmentNumberOfBits) > 0;
}

// Allocate memory in exponential distribution of size and random order
// of deallocations. Verify that allocation are within the arena and
// that allocator accounts for allocated and freed memory.
TEST(ConfigurableMemoryAllocator, ExponentialDistribution) {
  Logging::setMaxLoggingLevel(WARNING);
  VerboseLogging::setMaxLoggingLevel(0);

  const std::string arena1Name = "small-fixed-size";
  const size_t arena1BlockSize = (1 << alignmentNumberOfBits);
  const size_t arena1MaxAllocationSize = arena1BlockSize;
  const double arnea1RelativeSize = 0.2;

  const std::string arena2Name = "medium-flexible-size";
  const size_t arena2BlockSize = (1 << (alignmentNumberOfBits + 1));
  const size_t arena2MaxAllocationSize = arena1BlockSize * 2;
  const double arnea2RelativeSize = 0.2;

  const std::string arena3Name = "large-flexible-size";
  const size_t arena3BlockSize = (1 << (alignmentNumberOfBits + 2));
  ;
  const size_t arena3MaxAllocationSize = SIZE_MAX;
  const double arnea3RelativeSize = 0.6;

  const int nAllocationInterations = 100;
  const int nAllocations = 1000;
  const double multiplier = arena2MaxAllocationSize;
  const double perIterationFreeRatio = 0.8;
  void* arena = (void*)0x1000;
  const size_t arenaSizeInBytes = nAllocations * multiplier *
      (nAllocationInterations * (1.0 - perIterationFreeRatio));
  const double allocatedRatioJitThreshold = 0.9;

  std::random_device rd;
  std::mt19937 generator(rd());
  std::exponential_distribution<double> distribution(2.5);

  std::unique_ptr<MemoryAllocator> allocator = CreateMemoryAllocator(
      {"pool-and-2-freelists",
       alignmentNumberOfBits,
       {
           {arena1Name,
            arena1BlockSize,
            arena1MaxAllocationSize,
            arnea1RelativeSize,
            allocatedRatioJitThreshold},
           {arena2Name,
            arena2BlockSize,
            arena2MaxAllocationSize,
            arnea2RelativeSize,
            allocatedRatioJitThreshold},
           {arena3Name,
            arena3BlockSize,
            arena3MaxAllocationSize,
            arnea3RelativeSize,
            allocatedRatioJitThreshold},
       }},
      arena,
      arenaSizeInBytes);

  const MemoryAllocator::Stats initialStats = allocator->getStats();

  std::vector<void*> ptrs;
  for (int j = 0; j < nAllocationInterations; ++j) {
    for (int i = 0; i < nAllocations; ++i) {
      size_t bytesToAllocate =
          static_cast<size_t>(distribution(generator) * multiplier);

      const MemoryAllocator::Stats statsBeforeAlloc = allocator->getStats();
      void* ptr = allocator->allocate(bytesToAllocate);
      const MemoryAllocator::Stats statsAfterAlloc = allocator->getStats();

      VLOG(1) << "ptr=" << ptr << " bytesToAllocate=" << bytesToAllocate;
      if (bytesToAllocate == 0) {
        EXPECT_EQ(ptr, nullptr);
      } else {
        EXPECT_TRUE(isAligned(arena, ptr));

        // Verify that allocator account for alloacted memory.
        EXPECT_EQ(
            statsBeforeAlloc.statsInBytes.allocatedCount,
            statsAfterAlloc.statsInBytes.allocatedCount - bytesToAllocate);
        // verify that pointer is within the arena.
        EXPECT_GE(ptr, arena);
        EXPECT_LT(ptr, static_cast<char*>(arena) + arenaSizeInBytes);

        ptrs.push_back(ptr);
      }
    }

    std::random_shuffle(ptrs.begin(), ptrs.end());

    // Deallocate some of the allocated memory.
    std::vector<void*> unFreedPtrs;
    for (int i = 0; i < ptrs.size(); ++i) {
      if (i < (ptrs.size() * perIterationFreeRatio)) {
        const MemoryAllocator::Stats statsBeforeFree = allocator->getStats();
        allocator->free(ptrs[i]);
        const MemoryAllocator::Stats statsAfterFree = allocator->getStats();

        // Verify that allocator account for freed memory.
        EXPECT_GT(
            statsBeforeFree.statsInBytes.allocatedCount,
            statsAfterFree.statsInBytes.allocatedCount);
      } else {
        unFreedPtrs.push_back(ptrs[i]);
      }
    }
    ptrs.swap(unFreedPtrs);
  }

  // Free the leftover allocations.
  for (void* ptr : ptrs) {
    allocator->free(ptr);
  }

  const MemoryAllocator::Stats endStats = allocator->getStats();
  EXPECT_EQ(
      initialStats.statsInBytes.freeCount, endStats.statsInBytes.freeCount);
  EXPECT_EQ(endStats.statsInBytes.allocatedCount, 0);
}

} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
