/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/memory/managers/CachingMemoryManager.h"
#include <arrayfire.h> // Needed for af exception

#include <algorithm>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "flashlight/common/CppBackports.h"
namespace fl {

namespace {

constexpr size_t kMinBlockSize =
    512; // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576; // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer =
    2097152; // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer =
    20971520; // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc =
    10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152; // round up large allocs to 2 MiB

size_t roundSize(size_t size) {
  if (size < kMinBlockSize) {
    return kMinBlockSize;
  } else {
    return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
  }
}

size_t getAllocationSize(size_t size) {
  if (size <= kSmallSize) {
    return kSmallBuffer;
  } else if (size < kMinLargeAlloc) {
    return kLargeBuffer;
  } else {
    return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
  }
}

static bool BlockComparator(
    const CachingMemoryManager::Block* a,
    const CachingMemoryManager::Block* b) {
  if (a->size_ != b->size_) {
    return a->size_ < b->size_;
  }
  return (uintptr_t)a->ptr_ < (uintptr_t)b->ptr_;
}

} // namespace

CachingMemoryManager::DeviceMemoryInfo::DeviceMemoryInfo(int id)
    : deviceId_(id),
      largeBlocks_(BlockComparator),
      smallBlocks_(BlockComparator) {}

CachingMemoryManager::CachingMemoryManager(
    int numDevices,
    std::shared_ptr<MemoryManagerDeviceInterface> deviceInterface)
    : MemoryManagerAdapter(deviceInterface) {
  for (int i = 0; i < numDevices; ++i) {
    deviceMemInfos_.emplace(
        i, fl::cpp::make_unique<CachingMemoryManager::DeviceMemoryInfo>(i));
  }
}

void CachingMemoryManager::initialize() {}

void CachingMemoryManager::shutdown() {
  signalMemoryCleanup();
}

void CachingMemoryManager::addMemoryManagement(int device) {
  if (deviceMemInfos_.find(device) != deviceMemInfos_.end()) {
    return;
  }
  deviceMemInfos_.emplace(
      device,
      fl::cpp::make_unique<CachingMemoryManager::DeviceMemoryInfo>(device));
}

void CachingMemoryManager::removeMemoryManagement(int device) {
  if (deviceMemInfos_.find(device) == deviceMemInfos_.end()) {
    return;
  }
  deviceMemInfos_.erase(device);
}

void* CachingMemoryManager::alloc(
    bool userLock,
    const unsigned ndims,
    dim_t* dims,
    const unsigned elementSize) {
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);
  size_t size = elementSize;
  for (unsigned i = 0; i < ndims; ++i) {
    size *= dims[i];
  }
  size = roundSize(size);
  const bool isSmallAlloc = (size <= kSmallSize);
  CachingMemoryManager::Block searchKey(size);
  CachingMemoryManager::BlockSet& pool =
      isSmallAlloc ? memoryInfo.smallBlocks_ : memoryInfo.largeBlocks_;

  CachingMemoryManager::Block* block = nullptr;
  auto it = pool.lower_bound(&searchKey);
  if (it != pool.end()) {
    block = *it;
    pool.erase(it);
  } else {
    void* ptr = nullptr;
    size_t allocSize = getAllocationSize(size);
    mallocWithRetry(allocSize, &ptr); // could throw
    block = new Block(allocSize, ptr);
  }

  // If the block is larger than the requested size to handle another
  // allocation in the same large or small BlockSet, it will be split into two.
  // Note that we don't split a small stepsize out of a large one to keep the
  // implementation simple.
  CachingMemoryManager::Block* remaining = nullptr;
  size_t diff = block->size_ - size;
  if (diff >= (isSmallAlloc ? kMinBlockSize : kSmallSize)) {
    remaining = block;
    block = new Block(size, block->ptr_);
    block->prev_ = remaining->prev_;
    if (block->prev_) {
      block->prev_->next_ = block;
    }
    block->next_ = remaining;

    remaining->prev_ = block;
    remaining->ptr_ = static_cast<char*>(remaining->ptr_) + size;
    remaining->size_ -= size;
    pool.insert(remaining);
  }

  block->managerLock_ = !userLock;
  block->userLock_ = userLock;
  memoryInfo.allocatedBlocks_[block->ptr_] = block;
  return static_cast<void*>(block->ptr_);
}

size_t CachingMemoryManager::allocated(void* ptr) {
  if (!ptr) {
    return 0;
  }
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);
  auto it = memoryInfo.allocatedBlocks_.find(ptr);
  if (it == memoryInfo.allocatedBlocks_.end()) {
    return 0;
  }
  return (it->second)->size_;
}

void CachingMemoryManager::unlock(void* ptr, bool userUnlock) {
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);
  if (!ptr) {
    return;
  }
  auto it = memoryInfo.allocatedBlocks_.find(ptr);
  if (it == memoryInfo.allocatedBlocks_.end()) {
    // Probably came from user, just free it
    this->deviceInterface->nativeFree(ptr);
    ++memoryInfo.stats_.totalNativeFrees_;
    return;
  }

  CachingMemoryManager::Block* block = it->second;
  if (userUnlock) {
    block->userLock_ = false;
  } else {
    block->managerLock_ = false;
  }

  // Return early if either one is locked
  if (block->inUse()) {
    return;
  }
  memoryInfo.allocatedBlocks_.erase(it);
  freeBlock(block);
}

void CachingMemoryManager::freeBlock(CachingMemoryManager::Block* block) {
  if (block->inUse()) {
    throw std::runtime_error("trying to free a block which is in use");
  }
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);

  const bool isSmallAlloc = (block->size_ <= kSmallSize);
  CachingMemoryManager::BlockSet& pool =
      isSmallAlloc ? memoryInfo.smallBlocks_ : memoryInfo.largeBlocks_;
  tryMergeBlocks(block, block->prev_, pool);
  tryMergeBlocks(block, block->next_, pool);

  pool.insert(block);
}

/** combine previously split blocks */
void CachingMemoryManager::tryMergeBlocks(
    CachingMemoryManager::Block* dst,
    CachingMemoryManager::Block* src,
    BlockSet& pool) {
  if (!src || src->inUse()) {
    return;
  }
  if (dst->prev_ == src) {
    dst->ptr_ = src->ptr_;
    dst->prev_ = src->prev_;
    if (dst->prev_) {
      dst->prev_->next_ = dst;
    }
  } else {
    dst->next_ = src->next_;
    if (dst->next_) {
      dst->next_->prev_ = dst;
    }
  }
  dst->size_ += src->size_;
  pool.erase(src);
  delete src;
}

void CachingMemoryManager::mallocWithRetry(size_t size, void** ptr) {
  // Try nativeMalloc. If nativeMalloc fails, frees all non-split cached blocks
  // and retries.
  auto& memoryInfo = getDeviceMemoryInfo();
  try {
    ++memoryInfo.stats_.totalNativeMallocs_;
    *ptr = this->deviceInterface->nativeAlloc(size);
  } catch (std::exception& exUnused) {
    try {
      signalMemoryCleanup();
      ++memoryInfo.stats_.totalNativeMallocs_;
      *ptr = this->deviceInterface->nativeAlloc(size);
    } catch (std::exception& ex) {
      // note: af exception inherits from std exception
      std::cerr << "Unable to allocate memory with native alloc for size " +
              std::to_string(size) + " bytes with error '" + ex.what()
                << "'";
      // note: converting here an af exception to std exception prevents to
      // catch the af error code at the user level. Rethrowing.
      throw;
    }
  }
}

void CachingMemoryManager::freeBlocks(
    BlockSet& blocks,
    BlockSet::iterator it,
    BlockSet::iterator end) {
  // Frees all non-split blocks between `it` and `end`
  auto& memoryInfo = getDeviceMemoryInfo();
  while (it != end) {
    Block* block = *it;
    if (!block->isSplit()) {
      this->deviceInterface->nativeFree(static_cast<void*>(block->ptr_));
      ++memoryInfo.stats_.totalNativeFrees_;
      auto cur = it;
      ++it;
      blocks.erase(cur);
      delete block;
    } else {
      ++it;
    }
  }
}

void CachingMemoryManager::signalMemoryCleanup() {
  // Free all non-split cached blocks on device
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);

  freeBlocks(
      memoryInfo.largeBlocks_,
      memoryInfo.largeBlocks_.begin(),
      memoryInfo.largeBlocks_.end());

  freeBlocks(
      memoryInfo.smallBlocks_,
      memoryInfo.smallBlocks_.begin(),
      memoryInfo.smallBlocks_.end());
}

float CachingMemoryManager::getMemoryPressure() {
  return 0.0; // TODO: check if this is optimal
}

bool CachingMemoryManager::jitTreeExceedsMemoryPressure(size_t /* unused */) {
  return false; // TODO: check if this is optimal
}

void CachingMemoryManager::printInfo(const char* msg, const int /* unused */) {
  auto& memInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memInfo.mutexAll_);

  std::cout << msg << std::endl;
  std::cout << "MemoryManager type: CachingMemoryManager" << std::endl;
  std::cout << "Number of allocated blocks:" << memInfo.allocatedBlocks_.size()
            << std::endl;
  std::cout << "Size of free block pool (small):" << memInfo.smallBlocks_.size()
            << std::endl;
  std::cout << "Size of free block pool (large):" << memInfo.largeBlocks_.size()
            << std::endl;
  std::cout << "Total native mallocs:" << memInfo.stats_.totalNativeMallocs_
            << std::endl;
  std::cout << "Total native frees:" << memInfo.stats_.totalNativeFrees_
            << std::endl;
}

void CachingMemoryManager::userLock(const void* ptr) {
  if (!ptr) {
    return;
  }
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);

  auto it = memoryInfo.allocatedBlocks_.find(const_cast<void*>(ptr));
  if (it == memoryInfo.allocatedBlocks_.end()) {
    // Follows the behavior of DefaultMemoryManager
    auto block = new Block(kSmallBuffer, const_cast<void*>(ptr));
    block->managerLock_ = false;
    block->userLock_ = true;
    memoryInfo.allocatedBlocks_[block->ptr_] = block;
  } else {
    it->second->userLock_ = true;
  }
}

void CachingMemoryManager::userUnlock(const void* ptr) {
  this->unlock(const_cast<void*>(ptr), true);
}

bool CachingMemoryManager::isUserLocked(const void* ptr) {
  if (!ptr) {
    return false;
  }
  auto& memoryInfo = getDeviceMemoryInfo();
  std::lock_guard<std::recursive_mutex> lock(memoryInfo.mutexAll_);
  auto it = memoryInfo.allocatedBlocks_.find(const_cast<void*>(ptr));
  if (it == memoryInfo.allocatedBlocks_.end()) {
    return false;
  }
  return it->second->userLock_;
}

CachingMemoryManager::DeviceMemoryInfo&
CachingMemoryManager::getDeviceMemoryInfo(int device /* = -1*/) {
  if (device == -1) {
    device = this->deviceInterface->getActiveDeviceId();
  }
  auto it = deviceMemInfos_.find(device);
  if (it == deviceMemInfos_.end() || !it->second) {
    throw std::runtime_error("meminfo for the device doesn't exist");
  }
  return *(it->second);
}
} // namespace fl
