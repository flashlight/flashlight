/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "flashlight/flashlight/memory/MemoryManagerAdapter.h"
#include "flashlight/flashlight/memory/MemoryManagerDeviceInterface.h"

namespace fl {

/**
 * Reimplementation of CudaCachingAllocator from Torch adapted for flashlight
 * Sources :
 * https://github.com/torch/cutorch/blob/master/lib/THC/THCCachingAllocator.h
 * https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDACachingAllocator.cpp
 */
class CachingMemoryManager : public MemoryManagerAdapter {
 public:
  CachingMemoryManager(
      int numDevices,
      std::shared_ptr<MemoryManagerDeviceInterface> deviceInterface);
  ~CachingMemoryManager() override = default;
  void initialize() override;
  void shutdown() override;
  void* alloc(
      bool userLock,
      const unsigned ndims,
      dim_t* dims,
      const unsigned elSize) override;
  size_t allocated(void* ptr) override;
  void unlock(void* ptr, bool userLock) override;
  void printInfo(const char* msg, const int device) override;
  void userLock(const void* ptr) override;
  void userUnlock(const void* ptr) override;
  bool isUserLocked(const void* ptr) override;
  void signalMemoryCleanup() override;
  float getMemoryPressure() override;
  bool jitTreeExceedsMemoryPressure(size_t bytes) override;
  void addMemoryManagement(int device) override;
  void removeMemoryManagement(int device) override;

  // Block denotes a single allocated unit of memory.
  struct Block {
    size_t size_; // size of block in bytes
    void* ptr_; // memory address
    bool managerLock_; //  whether the memory is locked by the memory manager
    bool userLock_; // whether the memory is locked by the user
    Block* prev_; // prev block if split from a larger allocation
    Block* next_; // next block if split from a larger allocation

    bool isSplit() const {
      return (prev_ != nullptr) || (next_ != nullptr);
    }

    bool inUse() const {
      return managerLock_ || userLock_;
    }

    explicit Block(size_t size, void* ptr = nullptr)
        : size_(size),
          ptr_(ptr),
          managerLock_(false),
          userLock_(false),
          prev_(nullptr),
          next_(nullptr) {}
  };

  typedef bool (*Comparison)(const Block*, const Block*);
  typedef std::set<Block*, Comparison> BlockSet;

  // A structure to store allocation stats per device. A simple one for now.
  struct MemoryAllocationStats {
    size_t totalNativeMallocs_;
    size_t totalNativeFrees_;

    MemoryAllocationStats() : totalNativeMallocs_(0), totalNativeFrees_(0) {}
  };

  // Stores the mutex and misc variables per device so that we operate in a
  // thredsafe manner.
  struct DeviceMemoryInfo {
    int deviceId_;

    // lock around all operations
    std::recursive_mutex mutexAll_; // TODO:: improve perf using R/W locks

    // cached blocks larger than 1 MB
    BlockSet largeBlocks_;

    // cached blocks 1 MB or smaller
    BlockSet smallBlocks_;

    // allocated blocks by device pointer
    std::unordered_map<void*, Block*> allocatedBlocks_;

    MemoryAllocationStats stats_;

    explicit DeviceMemoryInfo(int id);
  };

 protected:
  std::unordered_map<int, std::unique_ptr<DeviceMemoryInfo>> deviceMemInfos_;

  CachingMemoryManager(const CachingMemoryManager& other) = delete;
  CachingMemoryManager(const CachingMemoryManager&& other) = delete;
  CachingMemoryManager& operator=(const CachingMemoryManager& other) = delete;
  CachingMemoryManager& operator=(const CachingMemoryManager&& other) = delete;

  // Returns the memory info of the caching allocator for the given device.
  // Using "-1" will return info for the current active device.
  DeviceMemoryInfo& getDeviceMemoryInfo(int device = -1);

  void
  freeBlocks(BlockSet& blocks, BlockSet::iterator it, BlockSet::iterator end);

  void mallocWithRetry(size_t size, void** ptr);

  void tryMergeBlocks(Block* dst, Block* src, BlockSet& freeBlocks);
  void freeBlock(Block* block);
};

} // namespace fl
