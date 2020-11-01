/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/experimental/memory/allocator/freelist/FreeList.h"

using namespace fl;
namespace {
constexpr double kEpsilon = 0.001;
const double kAllocatedRatioJitThreshold = 0.9;
const int kLogLevel = 1;

// Allocate memory in normal distribution of size and random order
// of deallocations. Verify that allocation are within the arena and
// that allocator accounts for allocated and freed memory.
TEST(FreeList, NormalDistribution) {
  Logging::setMaxLoggingLevel(WARNING);
  VerboseLogging::setMaxLoggingLevel(0);

  const int nAllocationInterations = 10; // Random value
  const int nAllocations = 1000; // Random large value
  // mean and stddev values should yield mostly positive values that are not too
  // much larger than arenaBlockSize
  const int mean = 500;
  const int stddev = 100;
  const double perIterationFreeRatio = 0.8; // Random 0..1 value.
  void* arena = (void*)0x1000;
  // arenaSizeInBytes is chosen to be just large enough to avoid OOM with
  // a small safety margin.
  const size_t arenaSizeInBytes = nAllocations * mean * nAllocationInterations;
  const size_t arenaBlockSize = 3; // Random "weird" value.

  FreeList allocator(
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
      const MemoryAllocator::Stats statsBeforeAlloc = allocator.getStats();
      const size_t bytesToAllocate = distribution(generator);
      void* ptr = allocator.allocate(bytesToAllocate);
      const MemoryAllocator::Stats statsAfterAlloc = allocator.getStats();

      // Verify that allocator account for alloacted memory.
      EXPECT_EQ(
          statsBeforeAlloc.statsInBytes.allocatedCount,
          statsAfterAlloc.statsInBytes.allocatedCount - bytesToAllocate);
      // verify that pointer is within the arena.
      EXPECT_GE(ptr, arena);
      EXPECT_LT(ptr, static_cast<char*>(arena) + arenaSizeInBytes);

      ptrs.push_back(ptr);
    }

    std::random_shuffle(ptrs.begin(), ptrs.end());

    // Deallocate some of the allocated memory.
    // Deallocate some of the allocated memory.
    std::vector<void*> unFreedPtrs;
    for (int i = 0; i < ptrs.size(); ++i) {
      if (i < (nAllocations * perIterationFreeRatio)) {
        const MemoryAllocator::Stats statsBeforeFree = allocator.getStats();
        allocator.free(ptrs[i]);
        // Verify that allocator account for freed memory.
        const MemoryAllocator::Stats statsAfterFree = allocator.getStats();
        EXPECT_GT(
            statsBeforeFree.statsInBytes.allocatedCount,
            statsAfterFree.statsInBytes.allocatedCount);
      } else {
        unFreedPtrs.push_back(ptrs[i]);
      }
    }
    ptrs.swap(unFreedPtrs);
    FL_VLOG(1) << allocator.prettyString();
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
TEST(FreeList, ExponentialDistribution) {
  Logging::setMaxLoggingLevel(WARNING);
  VerboseLogging::setMaxLoggingLevel(0);

  const int nAllocationInterations = 10; // Random value
  const int nAllocations = 10000; // Random large value
  const int multiplier = 500; // yields mostly positive values not too much
  // larger than arenaBlockSize.
  const double perIterationFreeRatio = 0.8;
  void* arena = (void*)0x1000;
  const size_t arenaBlockSize = 10; // Random value
  // arenaSizeInBytes is chosen to be just large enough to avoid OOM with
  // a small safety margin.
  const size_t arenaSizeInBytes =
      nAllocations * nAllocationInterations * multiplier * arenaBlockSize;

  FreeList allocator(
      "ExponentialDistribution",
      arena,
      arenaSizeInBytes,
      arenaBlockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  MemoryAllocator::Stats initialStats = allocator.getStats();

  std::random_device rd;
  std::mt19937 generator(rd());
  std::exponential_distribution<double> distribution(2.5);

  std::vector<void*> ptrs;
  for (int j = 0; j < nAllocationInterations; ++j) {
    for (int i = 0; i < nAllocations; ++i) {
      const MemoryAllocator::Stats statsBeforeAlloc = allocator.getStats();
      size_t bytesToAllocate = distribution(generator) * multiplier;
      void* ptr = allocator.allocate(bytesToAllocate);
      const MemoryAllocator::Stats statsAfterAlloc = allocator.getStats();

      if (bytesToAllocate == 0) {
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
      if (i < (nAllocations * perIterationFreeRatio)) {
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

TEST(FreeList, TooManyAllocs) {
  FreeList allocator(
      "TooManyAllocs",
      nullptr,
      100,
      10,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  EXPECT_THROW(while (true) { allocator.allocate(1); }, std::invalid_argument);
}

TEST(FreeList, AllocTooBig) {
  FreeList allocator(
      "AllocTooBig", nullptr, 100, 10, kAllocatedRatioJitThreshold, kLogLevel);
  EXPECT_THROW(
      while (true) { allocator.allocate(101); }, std::invalid_argument);
}

// Verify that internalFragmentationScore is 1-(bytes_asked_by_user /
// allocated_bytes)
TEST(FreeList, InternalFragmentation) {
  Logging::setMaxLoggingLevel(WARNING);
  VerboseLogging::setMaxLoggingLevel(0);

  void* arena = (void*)0x1000; // Random value
  const size_t blockSize = 10; // Random value
  const size_t arenaSize =
      blockSize * 2; // arenaSize is sufficient for two allocations.
  const double internalFragmentationScore1 =
      0.8; // blockSize*internalFragmentationScore1 should be an integer
  const size_t allocationSize1 =
      blockSize * (1.0 - internalFragmentationScore1 + kEpsilon);
  const double internalFragmentationScore2 =
      0.5; // blockSize*internalFragmentationScore2 should be an integer
  const size_t allocationSize2 =
      (1.0 - internalFragmentationScore2 + kEpsilon) * (2 * blockSize) -
      allocationSize1;

  FreeList allocator(
      "InternalFragmentation",
      arena,
      arenaSize,
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  const MemoryAllocator::Stats initialStats = allocator.getStats();
  FL_LOG(fl::INFO) << "initialStats=" << initialStats.prettyString();
  EXPECT_EQ(initialStats.internalFragmentationScore, 0.0);
  EXPECT_EQ(initialStats.externalFragmentationScore, 0.0);
  void* ptr1 = allocator.allocate(allocationSize1);
  const MemoryAllocator::Stats stats1 = allocator.getStats();
  FL_LOG(fl::INFO) << "stats1=" << stats1.prettyString();
  EXPECT_EQ(stats1.internalFragmentationScore, internalFragmentationScore1);
  EXPECT_EQ(stats1.externalFragmentationScore, 0.0);
  void* ptr2 = allocator.allocate(allocationSize2);
  const MemoryAllocator::Stats stats2 = allocator.getStats();
  FL_LOG(fl::INFO) << "stats2=" << stats2.prettyString();
  EXPECT_EQ(stats2.internalFragmentationScore, internalFragmentationScore2);
  EXPECT_EQ(stats2.externalFragmentationScore, 0.0);

  allocator.free(ptr1);
  allocator.free(ptr2);
  const MemoryAllocator::Stats endStats = allocator.getStats();
  EXPECT_EQ(endStats.internalFragmentationScore, 0.0);
  EXPECT_EQ(endStats.externalFragmentationScore, 0.0);
}

// Verify that externalFragmentationScore is 1-largest_chunk/free_chunks
TEST(FreeList, ExternalFragmentation) {
  void* arena = (void*)0x1000;
  const size_t blockSize = 1; // Random value
  const size_t arenaSize = blockSize * 5; // sufficient size for 5 allocations

  FreeList allocator(
      "ExternalFragmentation",
      arena,
      arenaSize,
      blockSize,
      kAllocatedRatioJitThreshold);
  const MemoryAllocator::Stats initialStats = allocator.getStats();
  FL_LOG(fl::INFO) << "initialStats=" << initialStats.prettyString();
  EXPECT_DOUBLE_EQ(initialStats.internalFragmentationScore, 0.0);
  EXPECT_DOUBLE_EQ(initialStats.externalFragmentationScore, 0.0);

  void* ptr0 = allocator.allocate(blockSize);
  void* ptr1 = allocator.allocate(blockSize);
  void* ptr2 = allocator.allocate(blockSize);
  void* ptr3 = allocator.allocate(blockSize);
  void* ptr4 = allocator.allocate(blockSize);

  // We should have now the following configuration (A-allocated, F-free):
  // A A A A A
  const MemoryAllocator::Stats afterAllocationStats = allocator.getStats();
  FL_LOG(fl::INFO) << "all allocated=" << initialStats.prettyString();
  EXPECT_DOUBLE_EQ(afterAllocationStats.internalFragmentationScore, 0.0);
  EXPECT_DOUBLE_EQ(afterAllocationStats.externalFragmentationScore, 0.0);

  allocator.free(ptr0);
  allocator.free(ptr2);
  // We should have now the following configuration (A-allocated, F-free):
  // F A F A A
  // 2 free blocks but max chunk size is 1.
  const double expectedExternalFragmentationScore1 = 1.0 - (1.0 / 2.0);
  const MemoryAllocator::Stats deAllocationStats1 = allocator.getStats();
  FL_LOG(fl::INFO) << "all allocated=" << deAllocationStats1.prettyString();
  EXPECT_DOUBLE_EQ(deAllocationStats1.internalFragmentationScore, 0.0);
  EXPECT_DOUBLE_EQ(
      deAllocationStats1.externalFragmentationScore,
      expectedExternalFragmentationScore1);

  allocator.free(ptr4);
  // We should have now the following configuration (A-allocated, F-free):
  // F A F A F
  // 3 free blocks but max chunk size is 1.
  const double expectedExternalFragmentationScore2 = 1.0 - (1.0 / 3.0);
  const MemoryAllocator::Stats deAllocationStats2 = allocator.getStats();
  FL_LOG(fl::INFO) << "all allocated=" << deAllocationStats2.prettyString();
  EXPECT_DOUBLE_EQ(deAllocationStats2.internalFragmentationScore, 0.0);
  EXPECT_DOUBLE_EQ(
      deAllocationStats2.externalFragmentationScore,
      expectedExternalFragmentationScore2);

  allocator.free(ptr1);
  // We should have now the following configuration (A-allocated, F-free):
  // F F F A F
  // 4 free blocks but max chunk size is 3.
  const double expectedExternalFragmentationScore3 = 1.0 - (3.0 / 4.0);
  const MemoryAllocator::Stats deAllocationStats3 = allocator.getStats();
  FL_LOG(fl::INFO) << "all allocated=" << deAllocationStats3.prettyString();
  EXPECT_DOUBLE_EQ(deAllocationStats3.internalFragmentationScore, 0.0);
  EXPECT_DOUBLE_EQ(
      deAllocationStats3.externalFragmentationScore,
      expectedExternalFragmentationScore3);

  allocator.free(ptr3);
  // We should have now the following configuration (A-allocated, F-free):
  // F F F F F
  // 5 free blocks with max chunk size of 5.
  const double expectedExternalFragmentationScore4 = 1.0 - (5.0 / 5.0);
  const MemoryAllocator::Stats deAllocationStats4 = allocator.getStats();
  FL_LOG(fl::INFO) << "all allocated=" << deAllocationStats4.prettyString();
  EXPECT_DOUBLE_EQ(deAllocationStats4.internalFragmentationScore, 0.0);
  EXPECT_DOUBLE_EQ(
      deAllocationStats4.externalFragmentationScore,
      expectedExternalFragmentationScore4);
}

} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
