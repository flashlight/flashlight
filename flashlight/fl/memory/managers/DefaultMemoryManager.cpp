/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/memory/managers/DefaultMemoryManager.h"

#include <af/defines.h>
#include <af/exception.h>
#include <af/memory.h>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

#include "flashlight/fl/memory/MemoryManagerDeviceInterface.h"

#define divup(a, b) (((a) + (b)-1) / (b))

namespace fl {

DefaultMemoryManager::MemoryInfo& DefaultMemoryManager::getCurrentMemoryInfo() {
  return memory[this->deviceInterface->getActiveDeviceId()];
}

void DefaultMemoryManager::cleanDeviceMemoryManager(int device) {
  if (this->debugMode) {
    return;
  }

  // This vector is used to store the pointers which will be deleted by
  // the memory manager. We are using this to avoid calling free while
  // the lock is being held because the CPU backend calls sync.
  std::vector<void*> freePtrs;
  size_t bytesFreed = 0;
  MemoryInfo& current = memory[device];
  {
    std::lock_guard<std::mutex> lock(this->memoryMutex);
    // Return if all buffers are locked
    if (current.totalBuffers == current.lockBuffers)
      return;
    freePtrs.reserve(current.freeMap.size());

    for (auto& kv : current.freeMap) {
      size_t numPtrs = kv.second.size();
      // Free memory by pushing the last element into the freePtrs
      // vector which will be freed once outside of the lock
      std::move(begin(kv.second), end(kv.second), std::back_inserter(freePtrs));
      current.totalBytes -= numPtrs * kv.first;
      bytesFreed += numPtrs * kv.first;
      current.totalBuffers -= numPtrs;
    }
    current.freeMap.clear();
  }

  std::stringstream ss;
  ss << "GC: Clearing " << freePtrs.size() << " buffers |"
     << std::to_string(bytesFreed) << " bytes";
  this->log(ss.str());

  // Free memory outside of the lock
  for (auto ptr : freePtrs) {
    this->deviceInterface->nativeFree(ptr);
  }
}

DefaultMemoryManager::DefaultMemoryManager(
    int numDevices,
    unsigned maxBuffers,
    bool debug,
    std::shared_ptr<MemoryManagerDeviceInterface> deviceInterface)
    : MemoryManagerAdapter(deviceInterface),
      memStepSize(1024),
      maxBuffers(maxBuffers),
      debugMode(debug),
      memory(numDevices) {
  // Check for environment variables
  // Debug mode
  if (const char* c = std::getenv("AF_MEM_DEBUG")) {
    this->debugMode = (std::string(c) != "0");
  }
  if (this->debugMode) {
    memStepSize = 1;
  }

  // Max Buffer count
  if (const char* c = std::getenv("AF_MAX_BUFFERS")) {
    this->maxBuffers = std::max(1, std::stoi(std::string(c)));
  }
}

void DefaultMemoryManager::initialize() {
  this->setMaxMemorySize();
}

void DefaultMemoryManager::shutdown() {
  signalMemoryCleanup();
}

void DefaultMemoryManager::addMemoryManagement(int device) {
  // If there is a memory manager allocated for this device id, we might
  // as well use it and the buffers allocated for it
  if (static_cast<size_t>(device) < memory.size())
    return;

  // Assuming, device need not be always the next device Lets resize to
  // current_size + device + 1 +1 is to account for device being 0-based
  // index of devices
  memory.resize(memory.size() + device + 1);
}

void DefaultMemoryManager::removeMemoryManagement(int device) {
  if ((size_t)device >= memory.size()) {
    throw std::runtime_error("No matching device found");
  }

  // Do garbage collection for the device and leave the
  // MemoryInfo struct from the memory vector intact
  cleanDeviceMemoryManager(device);
}

void DefaultMemoryManager::setMaxMemorySize() {
  for (unsigned n = 0; n < memory.size(); n++) {
    // Calls garbage collection when: totalBytes > memsize * 0.75 when
    // memsize < 4GB totalBytes > memsize - 1 GB when memsize >= 4GB If
    // memsize returned 0, then use 1GB
    size_t memsize = this->deviceInterface->getMaxMemorySize(n);
    memory[n].maxBytes = memsize == 0
        ? ONE_GB
        : std::max(memsize * 0.75, (double)(memsize - ONE_GB));
  }
}

void* DefaultMemoryManager::alloc(
    bool userLock,
    const unsigned ndims,
    dim_t* dims,
    const unsigned elementSize) {
  size_t bytes = elementSize;
  for (unsigned i = 0; i < ndims; ++i) {
    bytes *= dims[i];
  }

  void* ptr = nullptr;
  size_t allocBytes =
      this->debugMode ? bytes : (divup(bytes, memStepSize) * memStepSize);

  if (bytes > 0) {
    MemoryInfo& current = this->getCurrentMemoryInfo();
    LockedInfo info = {!userLock, userLock, allocBytes};

    // There is no memory cache in debug mode
    if (!this->debugMode) {
      // FIXME: Add better checks for garbage collection
      // Perhaps look at total memory available as a metric
      if (current.lockBytes >= current.maxBytes ||
          current.totalBuffers >= this->maxBuffers) {
        this->signalMemoryCleanup();
      }

      std::lock_guard<std::mutex> lock(this->memoryMutex);
      free_iter iter = current.freeMap.find(allocBytes);

      if (iter != current.freeMap.end() && !iter->second.empty()) {
        // Set to existing in from free map
        ptr = iter->second.back();
        iter->second.pop_back();
        current.lockedMap[ptr] = info;
        current.lockBytes += allocBytes;
        current.lockBuffers++;
      }
    }

    // Only comes here if buffer size not found or in debug mode
    if (ptr == nullptr) {
      // Perform garbage collection if memory can not be allocated
      try {
        ptr = this->deviceInterface->nativeAlloc(allocBytes);
      } catch (std::exception& ex) {
        // FIXME: assume that the exception is due to out of memory, and don't
        // continue propagating it
        // If out of memory, run garbage collect and try again
        // if (ex.err() != AF_ERR_NO_MEM) {
        //   throw;
        // }
        this->signalMemoryCleanup();
        ptr = this->deviceInterface->nativeAlloc(allocBytes);
      }
      std::lock_guard<std::mutex> lock(this->memoryMutex);
      // Increment these two only when it succeeds to come here.
      current.totalBytes += allocBytes;
      current.totalBuffers += 1;
      current.lockedMap[ptr] = info;
      current.lockBytes += allocBytes;
      current.lockBuffers++;
    }
  }
  return ptr;
}

size_t DefaultMemoryManager::allocated(void* ptr) {
  if (!ptr)
    return 0;
  MemoryInfo& current = this->getCurrentMemoryInfo();
  locked_iter iter = current.lockedMap.find((void*)ptr);
  if (iter == current.lockedMap.end())
    return 0;
  return (iter->second).bytes;
}

void DefaultMemoryManager::unlock(void* ptr, bool userUnlock) {
  // Shortcut for empty arrays
  if (!ptr) {
    return;
  }

  // Frees the pointer outside the lock.
  uptr_t freedPtr(
      nullptr, [this](void* p) { this->deviceInterface->nativeFree(p); });
  {
    std::lock_guard<std::mutex> lock(this->memoryMutex);
    MemoryInfo& current = this->getCurrentMemoryInfo();

    locked_iter iter = current.lockedMap.find((void*)ptr);

    // Pointer not found in locked map
    if (iter == current.lockedMap.end()) {
      // Probably came from user, just free it
      freedPtr.reset(ptr);
      return;
    }

    if (userUnlock) {
      (iter->second).userLock = false;
    } else {
      (iter->second).managerLock = false;
    }

    // Return early if either one is locked
    if ((iter->second).userLock || (iter->second).managerLock) {
      return;
    }

    size_t bytes = iter->second.bytes;
    current.lockBytes -= iter->second.bytes;
    current.lockBuffers--;

    if (this->debugMode) {
      // Just free memory in debug mode
      if ((iter->second).bytes > 0) {
        freedPtr.reset(iter->first);
        current.totalBuffers--;
        current.totalBytes -= iter->second.bytes;
      }
    } else {
      current.freeMap.at(bytes).emplace_back(ptr);
    }
    current.lockedMap.erase(iter);
  }
}

void DefaultMemoryManager::signalMemoryCleanup() {
  cleanDeviceMemoryManager(this->deviceInterface->getActiveDeviceId());
}

float DefaultMemoryManager::getMemoryPressure() {
  std::lock_guard<std::mutex> lock(this->memoryMutex);
  MemoryInfo& current = this->getCurrentMemoryInfo();
  if (current.lockBytes > current.maxBytes ||
      current.lockBuffers > maxBuffers) {
    return 1.0;
  } else {
    return 0.0;
  }
}

bool DefaultMemoryManager::jitTreeExceedsMemoryPressure(size_t bytes) {
  std::lock_guard<std::mutex> lock(this->memoryMutex);
  MemoryInfo& current = this->getCurrentMemoryInfo();
  return 2 * bytes > current.lockBytes;
}

void DefaultMemoryManager::printInfo(const char* msg, const int device) {
  const MemoryInfo& current = this->getCurrentMemoryInfo();

  printf("%s\n", msg);
  printf(
      "---------------------------------------------------------\n"
      "|     POINTER      |    SIZE    |  AF LOCK  | USER LOCK |\n"
      "---------------------------------------------------------\n");

  std::lock_guard<std::mutex> lock(this->memoryMutex);
  for (auto& kv : current.lockedMap) {
    const char* statusMngr = "Yes";
    const char* statusUser = "Unknown";
    if (kv.second.userLock)
      statusUser = "Yes";
    else
      statusUser = " No";

    const char* unit = "KB";
    double size = (double)(kv.second.bytes) / 1024;
    if (size >= 1024) {
      size = size / 1024;
      unit = "MB";
    }

    printf(
        "|  %14p  |  %6.f %s | %9s | %9s |\n",
        kv.first,
        size,
        unit,
        statusMngr,
        statusUser);
  }

  for (auto& kv : current.freeMap) {
    const char* statusMngr = "No";
    const char* statusUser = "No";

    const char* unit = "KB";
    double size = (double)(kv.first) / 1024;
    if (size >= 1024) {
      size = size / 1024;
      unit = "MB";
    }

    for (auto& ptr : kv.second) {
      printf(
          "|  %14p  |  %6.f %s | %9s | %9s |\n",
          ptr,
          size,
          unit,
          statusMngr,
          statusUser);
    }
  }

  printf("---------------------------------------------------------\n");
}

void DefaultMemoryManager::userLock(const void* ptr) {
  MemoryInfo& current = this->getCurrentMemoryInfo();

  std::lock_guard<std::mutex> lock(this->memoryMutex);

  locked_iter iter = current.lockedMap.find(const_cast<void*>(ptr));
  if (iter != current.lockedMap.end()) {
    iter->second.userLock = true;
  } else {
    LockedInfo info = {false, true, 100}; // This number is not relevant

    current.lockedMap[(void*)ptr] = info;
  }
}

void DefaultMemoryManager::userUnlock(const void* ptr) {
  this->unlock(const_cast<void*>(ptr), true);
}

bool DefaultMemoryManager::isUserLocked(const void* ptr) {
  MemoryInfo& current = this->getCurrentMemoryInfo();
  std::lock_guard<std::mutex> lock(this->memoryMutex);
  locked_iter iter = current.lockedMap.find(const_cast<void*>(ptr));
  if (iter != current.lockedMap.end()) {
    return iter->second.userLock;
  } else {
    return false;
  }
}

size_t DefaultMemoryManager::getMemStepSize() {
  std::lock_guard<std::mutex> lock(this->memoryMutex);
  return this->memStepSize;
}

void DefaultMemoryManager::setMemStepSize(size_t new_step_size) {
  std::lock_guard<std::mutex> lock(this->memoryMutex);
  this->memStepSize = new_step_size;
}

size_t DefaultMemoryManager::getMaxBytes() {
  std::lock_guard<std::mutex> lock(this->memoryMutex);
  return this->getCurrentMemoryInfo().maxBytes;
}

unsigned DefaultMemoryManager::getMaxBuffers() {
  return this->maxBuffers;
}

bool DefaultMemoryManager::checkMemoryLimit() {
  const MemoryInfo& current = this->getCurrentMemoryInfo();
  return current.lockBytes >= current.maxBytes ||
      current.totalBuffers >= this->maxBuffers;
}

} // namespace fl
