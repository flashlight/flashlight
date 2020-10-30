/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * The configurable memory allocator is obtained by calling:
 * std::unique_ptr<MemoryAllocator> CreateMemoryAllocator(config)
 * Config defines a a set of allocators assembled in a CompositeMemoryAllocator.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include "flashlight/fl/experimental/memory/allocator/MemoryAllocator.h"

namespace fl {

// Defines configuration for allocator that manages a portion of the full arena
// described in MemoryAllocatorConfiguration.
struct SubArenaConfiguration {
  std::string name_;
  size_t blockSize_;
  size_t maxAllocationSize_;
  // The portion of the arena that is given to this allocator entry is the sum
  // of all relativeSize over this relativeSize. The exact size of the allocator
  // is rounded to the nearest alignment.
  double relativeSize_;
  // Trigger ArrayFire JIT envalution in order to free memory when allocation
  // ratio (used memory/all memory) is greater than this threshold value.
  double allocatedRatioJitThreshold_;

  SubArenaConfiguration();
  SubArenaConfiguration(
      std::string name,
      size_t blockSize,
      size_t maxAllocationSize,
      double relativeSize,
      double allocatedRatioJitThreshold);
  std::string prettyString() const;

  // These methods are for support of the optimizer and only checks members that
  // we optimize.
  bool operator==(const SubArenaConfiguration& other) const;
  bool operator!=(const SubArenaConfiguration& other) const;
  bool operator<(const SubArenaConfiguration& other) const;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("name", name_),
       cereal::make_nvp("blockSize", blockSize_),
       cereal::make_nvp("maxAllocationSize", maxAllocationSize_),
       cereal::make_nvp("relativeSize", relativeSize_),
       cereal::make_nvp(
           "allocatedRatioJitThreshold", allocatedRatioJitThreshold_));
  }
};

// Define configuration composite memory allocator such that a large
// arena is broken to sub arenas, each managed by a separate allocator.
struct MemoryAllocatorConfiguration {
  std::string name_;
  // alignment is specified in number of bits, such that allocated pointer's
  // value is:
  // ptr = arena + n * (1 << alignmentNumberOfBits)
  size_t alignmentNumberOfBits_;
  std::vector<SubArenaConfiguration> subArenaConfiguration_;

  MemoryAllocatorConfiguration(
      std::string name,
      size_t alignmentNumberOfBits,
      std::vector<SubArenaConfiguration> subArenaConfiguration);
  MemoryAllocatorConfiguration();

  // Orders the sub arenas by maxAllocationSize, from small to big, sets the
  // last one to be catch all, that is, sets it to SIZE_MAX, and scales the
  // relativeSize of all sub arenas such that they sum up to 1.
  // Additionally, verifies that the sub arenas' block size a multiple of the
  // alignment. Throws an exception when it isn't.
  void normalize();

  std::string prettyString() const;
  size_t minBlockSize() const {
    return 1UL << alignmentNumberOfBits_;
  }

  bool operator==(const MemoryAllocatorConfiguration& other) const;
  bool operator!=(const MemoryAllocatorConfiguration& other) const;
  bool operator<(const MemoryAllocatorConfiguration& other) const;

  static MemoryAllocatorConfiguration loadJSon(std::istream& streamToConfig);
  void saveJSon(std::ostream& saveConfigStream) const;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("name", name_),
       cereal::make_nvp("subArenaConfiguration", subArenaConfiguration_),
       cereal::make_nvp("alignmentNumberOfBits", alignmentNumberOfBits_));
  }
};

// Returns a composite allocator configured by config.
std::unique_ptr<MemoryAllocator> CreateMemoryAllocator(
    MemoryAllocatorConfiguration config,
    void* arenaAddress,
    size_t arenaSizeInBytes,
    int logLevel = 1);

} // namespace fl
