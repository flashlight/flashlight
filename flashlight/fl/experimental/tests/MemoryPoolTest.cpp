/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/experimental/memory/allocator/memorypool/MemoryPool.h"

using namespace fl;

namespace {
constexpr double kEpsilon = 0.001;
const double kAllocatedRatioJitThreshold = 0.9;
const int kLogLevel = 1;

// Allocate memory in normal distribution of size and random order
// of deallocations. Verify that allocation are within the arena and
// that allocator accounts for allocated and freed memory.
TEST(MemoryPool, NormalDistribution) {
  Logging::setMaxLoggingLevel(WARNING);
  VerboseLogging::setMaxLoggingLevel(0);

  const int nAllocationInterations = 100; // Random value
  const int nAllocations = 1000; // Random value
  // means and stddev yield mostly values within valid range: 0 < val <
  // arenaBlockSize.
  const int mean = 5;
  const int stddev = 4;
  const double perIterationFreeRatio = 0.8; // Random 0..1 value
  const size_t arenaBlockSize = 10; // Random value
  // arenaSizeInBytes is chosen to be just large enough to avoid OOM with
  // a small safety margin.
  const size_t arenaSizeInBytes = arenaBlockSize * nAllocations;
  void* arena = (void*)0x1000;
  MemoryPool allocator(
      "NormalDistribution",
      arena,
      arenaSizeInBytes,
      arenaBlockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  const MemoryAllocator::Stats initialStats = allocator.getStats();

  std::minstd_rand0 generator;
  std::normal_distribution<double> distribution(mean, stddev);

  std::vector<void*> ptrs;
  for (int j = 0; j < nAllocationInterations; ++j) {
    for (int i = 0; i < nAllocations; ++i) {
      const size_t bytesToAllocate = distribution(generator);

      void* ptr = nullptr;
      const MemoryAllocator::Stats statsBeforeAlloc = allocator.getStats();
      if (bytesToAllocate <= arenaBlockSize) {
        ptr = allocator.allocate(bytesToAllocate);
      } else {
        EXPECT_THROW(
            { ptr = allocator.allocate(bytesToAllocate); },
            std::invalid_argument);
      }
      const MemoryAllocator::Stats statsAfterAlloc = allocator.getStats();

      if (bytesToAllocate == 0 || bytesToAllocate > arenaBlockSize) {
        EXPECT_EQ(ptr, nullptr);
      } else {
        EXPECT_NE(ptr, nullptr);
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
        const MemoryAllocator::Stats statsBeforeFree = allocator.getStats();
        allocator.free(ptrs[i]);
        const MemoryAllocator::Stats statsAfterFree = allocator.getStats();

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
    allocator.free(ptr);
  }

  const MemoryAllocator::Stats endStats = allocator.getStats();
  EXPECT_EQ(
      initialStats.statsInBytes.freeCount, endStats.statsInBytes.freeCount);
  EXPECT_EQ(endStats.statsInBytes.allocatedCount, 0);
}

// Allocate memory in exponential distribution of size and random order
// of deallocations. Verify that allocation are within the arena and
// that allocator accounts for allocated and freed memory.
TEST(MemoryPool, ExponentialDistribution) {
  Logging::setMaxLoggingLevel(WARNING);
  VerboseLogging::setMaxLoggingLevel(0);

  const int nAllocationInterations = 100; // Random value
  const int nAllocations = 1000; // Random large value
  const double multiplier = 2.5; // Chosen to yield mostly values within valid
  // range: 0 < val < arenaBlockSize.
  const double perIterationFreeRatio = 0.8; // Random 0..1 value
  const size_t arenaBlockSize = 10; // Random value
  void* arena = (void*)0x1000;
  // arenaSizeInBytes is chosen to be just large enough to avoid OOM with
  // a small safety margin.
  const size_t arenaSizeInBytes = nAllocations * arenaBlockSize;
  MemoryPool allocator(
      "ExponentialDistribution",
      arena,
      arenaSizeInBytes,
      arenaBlockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  const MemoryAllocator::Stats initialStats = allocator.getStats();

  std::random_device rd;
  std::mt19937 generator(rd());
  std::exponential_distribution<double> distribution(2.5);

  std::vector<void*> ptrs;
  for (int j = 0; j < nAllocationInterations; ++j) {
    for (int i = 0; i < nAllocations; ++i) {
      size_t bytesToAllocate = distribution(generator) * multiplier;

      void* ptr = nullptr;
      const MemoryAllocator::Stats statsBeforeAlloc = allocator.getStats();
      if (bytesToAllocate <= arenaBlockSize) {
        ptr = allocator.allocate(bytesToAllocate);
      } else {
        EXPECT_THROW(
            { ptr = allocator.allocate(bytesToAllocate); },
            std::invalid_argument);
      }
      const MemoryAllocator::Stats statsAfterAlloc = allocator.getStats();

      if (bytesToAllocate == 0 || bytesToAllocate > arenaBlockSize) {
        EXPECT_EQ(ptr, nullptr);
      } else {
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
        const MemoryAllocator::Stats statsBeforeFree = allocator.getStats();
        allocator.free(ptrs[i]);
        const MemoryAllocator::Stats statsAfterFree = allocator.getStats();

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
    allocator.free(ptr);
  }

  const MemoryAllocator::Stats endStats = allocator.getStats();
  EXPECT_EQ(
      initialStats.statsInBytes.freeCount, endStats.statsInBytes.freeCount);
  EXPECT_EQ(endStats.statsInBytes.allocatedCount, 0);
}

TEST(MemoryPool, TooManyAllocs) {
  fl::MemoryPool allocator(
      "TooManyAllocs",
      nullptr,
      100,
      10,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  EXPECT_THROW(while (true) { allocator.allocate(1); }, std::invalid_argument);
}

TEST(MemoryPool, AllocTooBig) {
  fl::MemoryPool allocator(
      "AllocTooBig", nullptr, 100, 10, kAllocatedRatioJitThreshold, kLogLevel);
  EXPECT_THROW(
      while (true) { allocator.allocate(101); }, std::invalid_argument);
}

// Verify that internalFragmentationScore is 1-(bytes_asked_by_user /
// allocated_bytes)
TEST(MemoryPool, Fragmentation) {
  Logging::setMaxLoggingLevel(WARNING);
  VerboseLogging::setMaxLoggingLevel(0);

  void* arena = (void*)0x1000;
  const size_t blockSize = 10; // Random value
  const size_t arenaSize = blockSize * 2; // sufficient for two allocations
  const double internalFragmentationScore1 =
      0.8; // internalFragmentationScore1*blockSize must be ineteger.
  const size_t allocationSize1 =
      blockSize * (1.0 - internalFragmentationScore1 + kEpsilon);
  const double internalFragmentationScore2 =
      0.5; // internalFragmentationScore2*blockSize must be ineteger.
  const size_t allocationSize2 =
      (1.0 - internalFragmentationScore2 + kEpsilon) * (2 * blockSize) -
      allocationSize1;

  fl::MemoryPool allocator(
      "Fragmentation",
      arena,
      arenaSize,
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  const MemoryAllocator::Stats initialStats = allocator.getStats();
  FL_LOG(fl::INFO) << "initialStats=" << initialStats.prettyString();
  EXPECT_EQ(initialStats.internalFragmentationScore, 0.0);
  EXPECT_EQ(initialStats.externalFragmentationScore, 0.0);
  allocator.allocate(allocationSize1);
  const MemoryAllocator::Stats stats1 = allocator.getStats();
  FL_LOG(fl::INFO) << "stats1=" << stats1.prettyString();
  // 2 bytes allocated out
  EXPECT_EQ(stats1.internalFragmentationScore, internalFragmentationScore1);
  EXPECT_EQ(stats1.externalFragmentationScore, 0.0);
  allocator.allocate(allocationSize2);
  const MemoryAllocator::Stats stats2 = allocator.getStats();
  FL_LOG(fl::INFO) << "stats2=" << stats2.prettyString();
  EXPECT_EQ(stats2.internalFragmentationScore, internalFragmentationScore2);
  EXPECT_EQ(stats2.externalFragmentationScore, 0.0);
}

} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
