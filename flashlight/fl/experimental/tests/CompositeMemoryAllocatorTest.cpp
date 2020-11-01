/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/fl/common/CppBackports.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/experimental/memory/allocator/CompositeMemoryAllocator.h"
#include "flashlight/fl/experimental/memory/allocator/freelist/FreeList.h"

using namespace fl;

namespace {
const double kAllocatedRatioJitThreshold = 0.9;
const int kLogLevel = 1;

// Loop over a test that allocates elements and frees half of them in random
// order.
TEST(FreeList, MaxAllocationSizeRepeatedly) {
  Logging::setMaxLoggingLevel(ERROR);
  VerboseLogging::setMaxLoggingLevel(0);

  const size_t nIterations = 100; // Random value
  const size_t allocCntPerIteration = 20; // Random value
  // Addresses are far enough from each other to allow mutually exclusive
  // allocation values
  void* address1 = (void*)0x10000;
  void* address2 = (void*)0x20000;
  void* address3 = (void*)0x30000;
  const size_t blockSize = 1; // Random value
  // Strictly increasing multiples of blockSize.
  const size_t maxAllocationSize1 = 2 * blockSize;
  const size_t maxAllocationSize2 = 10 * blockSize;
  const size_t maxAllocationSize3 = 100 * blockSize;
  const size_t numAllocations = nIterations * allocCntPerIteration;

  auto freelist1 = fl::cpp::make_unique<FreeList>(
      "small-freelist",
      address1,
      numAllocations * maxAllocationSize1,
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  auto freelist2 = fl::cpp::make_unique<FreeList>(
      "medium-freelist",
      address2,
      numAllocations * maxAllocationSize2,
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  auto freelist3 = fl::cpp::make_unique<FreeList>(
      "large-freelist",
      address3,
      numAllocations * maxAllocationSize3,
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);

  fl::CompositeMemoryAllocator allocator("3-freelists");
  allocator.add({maxAllocationSize1, std::move(freelist1)});
  allocator.add({maxAllocationSize2, std::move(freelist2)});
  allocator.add({maxAllocationSize3, std::move(freelist3)});

  const MemoryAllocator::Stats initialStats = allocator.getStats();

  std::vector<void*> ptrs;

  for (int i = 0; i < nIterations; ++i) {
    for (int j = 0; j < allocCntPerIteration; ++j) {
      ptrs.push_back(allocator.allocate(maxAllocationSize1));
      ptrs.push_back(allocator.allocate(maxAllocationSize2));
      ptrs.push_back(allocator.allocate(maxAllocationSize3));
    }

    std::random_shuffle(ptrs.begin(), ptrs.end());

    for (int j = 0; j < (ptrs.size() / 2); ++j) {
      allocator.free(ptrs.back());
      ptrs.pop_back();
    }
  }

  for (void* ptr : ptrs) {
    allocator.free(ptr);
  }

  const MemoryAllocator::Stats endStats = allocator.getStats();
  EXPECT_EQ(
      initialStats.statsInBytes.freeCount, endStats.statsInBytes.freeCount);
  EXPECT_EQ(endStats.statsInBytes.allocatedCount, 0);
}

// Create composite allocator with sufficient memory for all allocation but
// insufficient memory in smallest buckets such that small allocation are
// propagated to larger buckets. Allocate memory in exponential distribution of
// size and random order of deallocations. Verify that allocation are within the
// arena and that allocator accounts for allocated and freed memory.
TEST(FreeList, ExponentialDistribution) {
  Logging::setMaxLoggingLevel(ERROR);
  VerboseLogging::setMaxLoggingLevel(0);

  const size_t nIterations = 100; // Random value
  const size_t allocCntPerIteration = 20; // Random value
  // Addresses are far enough from each other to allow mutually exclusive
  // allocation values
  void* address1 = (void*)0x10000;
  void* address2 = (void*)0x20000;
  void* address3 = (void*)0x30000;
  const size_t blockSize = 2; // Random value
  // Strictly increasing multiples of blockSize.
  const size_t maxAllocationSize1 = 2 * blockSize;
  const size_t maxAllocationSize2 = 10 * blockSize;
  const size_t maxAllocationSize3 = SIZE_MAX; // Catch all size
  const double perIterationFreeRatio = 0.8; // Random 0..1 value
  const size_t numAllocations =
      nIterations * allocCntPerIteration * (1.0 - perIterationFreeRatio);
  const int multiplier = maxAllocationSize2; // yields values that fall within
  // all three allocators.
  auto freelist1 = fl::cpp::make_unique<FreeList>(
      "small-freelist",
      address1,
      numAllocations * maxAllocationSize1,
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  auto freelist2 = fl::cpp::make_unique<FreeList>(
      "medium-freelist",
      address2,
      numAllocations * maxAllocationSize2,
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  auto freelist3 = fl::cpp::make_unique<FreeList>(
      "large-freelist",
      address3,
      numAllocations * maxAllocationSize2 *
          2, // This size seems to be just enough.
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);

  fl::CompositeMemoryAllocator allocator("3-freelists");
  allocator.add({maxAllocationSize1, std::move(freelist1)});
  allocator.add({maxAllocationSize2, std::move(freelist2)});
  allocator.add({maxAllocationSize3, std::move(freelist3)});

  const MemoryAllocator::Stats initialStats = allocator.getStats();

  std::random_device rd;
  std::mt19937 generator(rd());
  std::exponential_distribution<double> distribution(50);

  std::vector<void*> ptrs;
  for (int i = 0; i < nIterations; ++i) {
    for (int j = 0; j < allocCntPerIteration; ++j) {
      const size_t bytesToAllocate = distribution(generator) * multiplier +
          1; // add 1 to avoid majority of values be zero.

      const MemoryAllocator::Stats statsBeforeAlloc = allocator.getStats();
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
        EXPECT_GE(ptr, address1);
        ptrs.push_back(ptr);
      }
    }

    std::random_shuffle(ptrs.begin(), ptrs.end());

    // Deallocate some of the allocated memory.
    std::vector<void*> unFreedPtrs;
    for (int k = 0; k < ptrs.size(); ++k) {
      if (k < (ptrs.size() * perIterationFreeRatio)) {
        const MemoryAllocator::Stats statsBeforeFree = allocator.getStats();
        allocator.free(ptrs[k]);
        const MemoryAllocator::Stats statsAfterFree = allocator.getStats();

        // Verify that allocator account for freed memory.
        EXPECT_GT(
            statsBeforeFree.statsInBytes.allocatedCount,
            statsAfterFree.statsInBytes.allocatedCount);
      } else {
        unFreedPtrs.push_back(ptrs[k]);
      }
    }
    ptrs.swap(unFreedPtrs);
  }

  for (void* ptr : ptrs) {
    allocator.free(ptr);
  }

  const MemoryAllocator::Stats endStats = allocator.getStats();
  EXPECT_EQ(
      initialStats.statsInBytes.freeCount, endStats.statsInBytes.freeCount);
  EXPECT_EQ(endStats.statsInBytes.allocatedCount, 0);
}

// Verify that internalFragmentationScore is weighted more heavily for allocator
// with larger arenas.
// Verify that externalFragmentationScore is unaffected by
// internalFragmentationScore.
TEST(FreeList, InternalFragmentation) {
  Logging::setMaxLoggingLevel(ERROR);
  VerboseLogging::setMaxLoggingLevel(0);

  // Addresses are far enough from each other to allow mutually exclusive
  // allocation values
  void* address1 = (void*)0x10000;
  void* address2 = (void*)0x20000;
  void* address3 = (void*)0x30000;
  const size_t blockSize = 2; // Random value
  // Strictly increasing multiples of blockSize.
  const size_t maxAllocationSize1 = 2 * blockSize;
  const size_t maxAllocationSize2 = 10 * blockSize;
  const size_t maxAllocationSize3 = SIZE_MAX; // Catch all size
  const double partOfBlockToAllocate = 0.5;
  // Size that matches from allocator1 with some internal fragmentation.
  const size_t alloc1Size = blockSize * partOfBlockToAllocate;
  // Size that matches from allocator2 with some internal fragmentation.
  const size_t alloc2Size =
      maxAllocationSize1 + blockSize * partOfBlockToAllocate;
  // Size that matches from allocator3 with some internal fragmentation.
  const size_t alloc3Size =
      maxAllocationSize2 + blockSize * partOfBlockToAllocate;

  auto freelist1 = fl::cpp::make_unique<FreeList>(
      "small-freelist",
      address1,
      maxAllocationSize1,
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  auto freelist2 = fl::cpp::make_unique<FreeList>(
      "medium-freelist",
      address2,
      maxAllocationSize2,
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  auto freelist3 = fl::cpp::make_unique<FreeList>(
      "large-freelist",
      address3,
      maxAllocationSize2 * 2, // This size seems to be just enough.
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);

  fl::CompositeMemoryAllocator allocator("3-freelists");
  allocator.add({maxAllocationSize1, std::move(freelist1)});
  allocator.add({maxAllocationSize2, std::move(freelist2)});
  allocator.add({maxAllocationSize3, std::move(freelist3)});

  const MemoryAllocator::Stats initialStats = allocator.getStats();
  FL_LOG(fl::INFO) << "initialStats=" << initialStats.prettyString();
  EXPECT_EQ(initialStats.internalFragmentationScore, 0.0);
  EXPECT_EQ(initialStats.externalFragmentationScore, 0.0);
  void* ptr1 = allocator.allocate(alloc1Size);
  const MemoryAllocator::Stats stats1 = allocator.getStats();
  FL_LOG(fl::INFO) << "stats1=" << stats1.prettyString();
  EXPECT_GT(
      stats1.internalFragmentationScore,
      initialStats.internalFragmentationScore);
  EXPECT_EQ(stats1.externalFragmentationScore, 0.0);

  void* ptr2 = allocator.allocate(alloc2Size);
  const MemoryAllocator::Stats stats2 = allocator.getStats();
  FL_LOG(fl::INFO) << "stats2=" << stats2.prettyString();
  EXPECT_GT(
      stats2.internalFragmentationScore, stats1.internalFragmentationScore);
  EXPECT_EQ(stats2.externalFragmentationScore, 0.0);

  void* ptr3 = allocator.allocate(alloc3Size);
  const MemoryAllocator::Stats stats3 = allocator.getStats();
  FL_LOG(fl::INFO) << "stats2=" << stats2.prettyString();
  EXPECT_GT(
      stats3.internalFragmentationScore, stats2.internalFragmentationScore);
  EXPECT_EQ(stats3.externalFragmentationScore, 0.0);

  allocator.free(ptr1);
  allocator.free(ptr2);
  allocator.free(ptr3);

  const MemoryAllocator::Stats endStats = allocator.getStats();
  EXPECT_EQ(
      initialStats.statsInBytes.freeCount, endStats.statsInBytes.freeCount);
  EXPECT_EQ(endStats.statsInBytes.allocatedCount, 0);
}

// Verify that externalFragmentationScore is weighted more heavily for allocator
// with larger arenas.
// Verify that internalFragmentationScore is unaffected by
// externalFragmentationScore.
TEST(FreeList, ExternalFragmentation) {
  Logging::setMaxLoggingLevel(INFO);
  VerboseLogging::setMaxLoggingLevel(0);

  // Addresses are far enough from each other to allow mutually exclusive
  // allocation values
  void* address1 = (void*)0x10000;
  void* address2 = (void*)0x20000;
  void* address3 = (void*)0x30000;
  const size_t blockSize = 2; // Random value
  const size_t numAllocations = 3;
  // numberBlocksPerAllocator allows extra space for fragmentation.
  const size_t numberBlocksPerAllocator = numAllocations + 2;
  // Strictly increasing multiples of blockSize.
  const size_t maxAllocationSize1 = 2 * blockSize;
  const size_t maxAllocationSize2 = 10 * blockSize;
  const size_t maxAllocationSize3 = SIZE_MAX; // Catch all size

  auto freelist1 = fl::cpp::make_unique<FreeList>(
      "small-freelist",
      address1,
      maxAllocationSize1 * numberBlocksPerAllocator,
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  auto freelist2 = fl::cpp::make_unique<FreeList>(
      "medium-freelist",
      address2,
      maxAllocationSize2 * numberBlocksPerAllocator,
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);
  auto freelist3 = fl::cpp::make_unique<FreeList>(
      "large-freelist",
      address3,
      maxAllocationSize2 * numberBlocksPerAllocator *
          2, // This size seems to be just enough.
      blockSize,
      kAllocatedRatioJitThreshold,
      kLogLevel);

  fl::CompositeMemoryAllocator allocator("3-freelists");
  allocator.add({maxAllocationSize1, std::move(freelist1)});
  allocator.add({maxAllocationSize2, std::move(freelist2)});
  allocator.add({maxAllocationSize3, std::move(freelist3)});

  const MemoryAllocator::Stats initialStats = allocator.getStats();
  FL_LOG(fl::INFO) << "initialStats=" << initialStats.prettyString();
  FL_VLOG(1) << "allocator=" << allocator.prettyString();
  EXPECT_EQ(initialStats.internalFragmentationScore, 0.0);
  EXPECT_EQ(initialStats.externalFragmentationScore, 0.0);

  // Alocate from freelist1
  void* ptr1 = allocator.allocate(maxAllocationSize1);
  void* ptr2 = allocator.allocate(maxAllocationSize1);
  void* ptr3 = allocator.allocate(maxAllocationSize1);
  allocator.free(ptr2);
  // We should have now the following configuration (A-allocated, F-free):
  // A F A F F
  const MemoryAllocator::Stats stats1 = allocator.getStats();
  FL_LOG(fl::INFO) << "stats1=" << stats1.prettyString();
  FL_VLOG(1) << "allocator=" << allocator.prettyString();
  EXPECT_EQ(stats1.internalFragmentationScore, 0.0);
  EXPECT_GT(
      stats1.externalFragmentationScore,
      initialStats.externalFragmentationScore);

  // Alocate from freelist2
  void* ptr4 = allocator.allocate(maxAllocationSize2);
  void* ptr5 = allocator.allocate(maxAllocationSize2);
  void* ptr6 = allocator.allocate(maxAllocationSize2);
  allocator.free(ptr5);
  // We should have now the following configuration (A-allocated, F-free):
  // A F A F F
  const MemoryAllocator::Stats stats2 = allocator.getStats();
  FL_LOG(fl::INFO) << "stats2=" << stats2.prettyString();
  FL_VLOG(1) << "allocator=" << allocator.prettyString();
  EXPECT_EQ(stats1.internalFragmentationScore, 0.0);
  EXPECT_GT(
      stats2.externalFragmentationScore, stats1.externalFragmentationScore);

  // Alocate from freelist3
  void* ptr7 = allocator.allocate(maxAllocationSize2 * 2);
  void* ptr8 = allocator.allocate(maxAllocationSize2 * 2);
  void* ptr9 = allocator.allocate(maxAllocationSize2 * 2);
  allocator.free(ptr8);
  // We should have now the following configuration (A-allocated, F-free):
  // A F A F F
  const MemoryAllocator::Stats stats3 = allocator.getStats();
  FL_LOG(fl::INFO) << "stats3=" << stats3.prettyString();
  FL_VLOG(1) << "allocator=" << allocator.prettyString();
  EXPECT_EQ(stats1.internalFragmentationScore, 0.0);
  EXPECT_GT(
      stats3.externalFragmentationScore, stats2.externalFragmentationScore);

  // We free more memory, but largest avaialable chunks remain unchanged, this
  // leads to higher external fragmentation.
  allocator.free(ptr1);
  allocator.free(ptr4);
  allocator.free(ptr7);
  // We should have now the following configuration for all allocators
  // (A-allocated, F-free):
  // F F A F F
  const MemoryAllocator::Stats stats4 = allocator.getStats();
  FL_LOG(fl::INFO) << "stats4=" << stats4.prettyString();
  FL_VLOG(1) << "allocator=" << allocator.prettyString();
  EXPECT_EQ(stats1.internalFragmentationScore, 0.0);
  EXPECT_GT(
      stats4.externalFragmentationScore, stats3.externalFragmentationScore);

  allocator.free(ptr3);
  allocator.free(ptr6);
  allocator.free(ptr9);
  // All memory is freed.
  const MemoryAllocator::Stats endStats = allocator.getStats();
  FL_LOG(fl::INFO) << "endStats=" << initialStats.prettyString();
  FL_VLOG(1) << "allocator=" << allocator.prettyString();
  EXPECT_EQ(initialStats.internalFragmentationScore, 0.0);
  EXPECT_EQ(initialStats.externalFragmentationScore, 0.0);
}

} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
