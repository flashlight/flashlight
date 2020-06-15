/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace fl {

constexpr double kJitTreeExceedsMemoryPressureAllocatedRatioThreshold = 0.7;
constexpr size_t kHistogramBucketCountPrettyString = 15;

// Memory allocator base class. Supports allocation, deallocation, stats
// collections, and giving information about allocated pointers.
class MemoryAllocator {
 public:
  explicit MemoryAllocator(std::string name, int logLevel = 1);
  virtual ~MemoryAllocator() = default;

  virtual void* allocate(size_t size) = 0;
  virtual void free(void* ptr) = 0;

  // Helper for Stats
  struct CommonStats {
    size_t arenaSize;
    size_t freeCount;
    size_t allocatedCount;
    size_t maxAllocatedCount;

    CommonStats();

    void allocate(size_t n);
    void free(size_t n);

    double allocatedRatio() const; // allocatedCount/arenaSize
    double maxAllocatedRatio() const; // maxAllocatedCount/arenaSize
    bool operator==(const CommonStats& other) const;
    bool operator!=(const CommonStats& other) const;
    // Reports diff when *this!=other otherwise return an empty string.
    std::string diffPrettyString(const CommonStats& other) const;
    std::string prettyString() const;
  };

  struct Stats {
    std::string allocatorName;
    void* arena;
    size_t blockSize;
    size_t allocationsCount; // number of calls to allocate()
    size_t deAllocationsCount; // number of calls to free()
    double internalFragmentationScore; // 0..1  no-frag .. highly-frag
    double externalFragmentationScore; // 0..1  no-frag .. highly-frag
    double maxInternalFragmentationScore; // 0..1  no-frag .. highly-frag
    double maxExternalFragmentationScore; // 0..1  no-frag .. highly-frag
    size_t oomEventCount; // Internal OOM unvisible to user.
    // Indication of how many work steps performed by this allocator.
    unsigned long long performanceCost;

    // CompositeMemoryAllocator stats.
    bool failToAllocate; // User visible OOM
    std::vector<Stats> subArenaStats; // Stats per internal sub arena.

    CommonStats statsInBytes;
    CommonStats statsInBlocks;

    Stats();
    Stats(void* arena, size_t arenaSizeInBytes, size_t blockSize);

    void incrementAllocationsCount();
    void incrementDeAllocationsCount();
    void setExternalFragmentationScore(double score);
    void incrementOomEventCount();
    void allocate(size_t bytes, size_t blocks);
    void free(size_t bytes, size_t blocks);
    void addPerformanceCost(size_t cost);

    // arena value is not compared in this operator.
    bool operator==(const Stats& other) const;
    bool operator!=(const Stats& other) const;
    // Reports diff when *this!=other otherwise return an empty string.
    std::string diffPrettyString(const Stats& other) const;
    std::string prettyString() const;
  };
  virtual Stats getStats() const = 0;

  virtual size_t getAllocatedSizeInBytes(void* ptr) const = 0;

  virtual std::string prettyString() const = 0;

  void setName(std::string name);
  const std::string& getName() const;
  int getLogLevel() const;
  virtual void setLogLevel(int logLevel);
  virtual bool jitTreeExceedsMemoryPressure(size_t bytes) = 0;

 protected:
  std::string name_;
  int logLevel_;
};

} // namespace fl
