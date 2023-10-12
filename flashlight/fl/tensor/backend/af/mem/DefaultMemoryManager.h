/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/backend/af/mem/MemoryManagerAdapter.h"

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace fl {

/**
 * A clone of ArrayFire's default memory manager that inherits from
 * `MemoryManagerAdapter` and can be used with flashlight abstractions so as to
 * facilitate logging and inspection of internal memory manager state during
 * runs.
 *
 * Additionally provides a simple starting point for other memory manager
 * implementations.
 */
class DefaultMemoryManager : public MemoryManagerAdapter {
  constexpr static unsigned MAX_BUFFERS = 1000;
  constexpr static size_t ONE_GB = 1 << 30;

  struct LockedInfo {
    bool managerLock;
    bool userLock;
    size_t bytes;
  };

  using locked_t = typename std::unordered_map<void*, LockedInfo>;
  using locked_iter = typename locked_t::iterator;

  using free_t = std::unordered_map<size_t, std::vector<void*>>;
  using free_iter = typename free_t::iterator;

  using uptr_t = std::unique_ptr<void, std::function<void(void*)>>;

  struct MemoryInfo {
    locked_t lockedMap;
    free_t freeMap;

    size_t lockBytes;
    size_t lockBuffers;
    size_t totalBytes;
    size_t totalBuffers;
    size_t maxBytes;

    MemoryInfo()
        // Calling getMaxMemorySize() here calls the virtual function
        // that returns 0 Call it from outside the constructor.
        : lockBytes(0),
          lockBuffers(0),
          totalBytes(0),
          totalBuffers(0),
          maxBytes(ONE_GB) {}

    MemoryInfo(MemoryInfo& other) = delete;
    MemoryInfo(MemoryInfo&& other) = default;
    MemoryInfo& operator=(MemoryInfo& other) = delete;
    MemoryInfo& operator=(MemoryInfo&& other) = default;
  };

  size_t memStepSize;
  unsigned maxBuffers;

  bool debugMode;

  MemoryInfo& getCurrentMemoryInfo();

 public:
  DefaultMemoryManager(
      int numDevices,
      unsigned maxBuffers,
      bool debug,
      std::shared_ptr<MemoryManagerDeviceInterface> deviceInterface);
  ~DefaultMemoryManager() = default;
  void initialize() override;
  void shutdown() override;
  void* alloc(
      bool userLock,
      const unsigned ndims,
      dim_t* dims,
      const unsigned elSize) override;
  size_t allocated(void* ptr) override;
  void unlock(void* ptr, bool userLock) override;
  void printInfo(
      const char* msg,
      const int device,
      std::ostream* ostream = &std::cout) override;
  void userLock(const void* ptr) override;
  void userUnlock(const void* ptr) override;
  bool isUserLocked(const void* ptr) override;
  void signalMemoryCleanup() override;
  float getMemoryPressure() override;
  bool jitTreeExceedsMemoryPressure(size_t bytes) override;
  void addMemoryManagement(int device) override;
  void removeMemoryManagement(int device) override;
  // Implementation-specific functions
  void setMaxMemorySize();
  size_t getMemStepSize() override;
  void setMemStepSize(size_t size) override;
  size_t getMaxBytes();
  unsigned getMaxBuffers();
  bool checkMemoryLimit();

 protected:
  DefaultMemoryManager(const DefaultMemoryManager& other) = delete;
  DefaultMemoryManager(const DefaultMemoryManager&& other) = delete;
  DefaultMemoryManager& operator=(const DefaultMemoryManager& other) = delete;
  DefaultMemoryManager& operator=(const DefaultMemoryManager&& other) = delete;

  std::mutex memoryMutex;
  // backend-specific
  std::vector<MemoryInfo> memory;
  // backend-agnostic
  void cleanDeviceMemoryManager(int device);
};

} // namespace fl
