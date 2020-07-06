/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/memory/MemoryManagerInstaller.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "flashlight/common/Logging.h"
#include "flashlight/common/Utils.h"
#include "flashlight/memory/managers/CachingMemoryManager.h"

namespace fl {

// Statics from MemoryManagerInstaller
std::once_flag MemoryManagerInstaller::startupMemoryInitialize_;
std::shared_ptr<MemoryManagerInstaller>
    MemoryManagerInstaller::startupMemoryManagerInstaller_;
std::shared_ptr<MemoryManagerAdapter>
    MemoryManagerInstaller::currentlyInstalledMemoryManager_;

namespace {

bool init = MemoryManagerInstaller::installDefaultMemoryManager();

} // namespace

MemoryManagerAdapter* MemoryManagerInstaller::getImpl(
    af_memory_manager manager) {
  void* ptr;
  AF_CHECK(af_memory_manager_get_payload(manager, &ptr));
  return (MemoryManagerAdapter*)ptr;
}

MemoryManagerInstaller::MemoryManagerInstaller(
    std::shared_ptr<MemoryManagerAdapter> managerImpl)
    : impl_(managerImpl) {
  if (!impl_) {
    throw std::invalid_argument(
        "MemoryManagerInstaller::MemoryManagerInstaller - "
        "passed MemoryManagerAdapter is null");
  }

  af_memory_manager itf = impl_->getHandle();
  if (!impl_->getHandle()) {
    throw std::invalid_argument(
        "MemoryManagerInstaller::MemoryManagerInstaller - "
        "passed MemoryManagerAdapter has null handle");
  }

  // Set appropriate function pointers for each class method
  auto initializeFn = [](af_memory_manager manager) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("initialize");
    m->initialize();
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_initialize_fn(itf, initializeFn));
  auto shutdownFn = [](af_memory_manager manager) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("shutdown");
    m->shutdown();
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_shutdown_fn(itf, shutdownFn));
  // ArrayFire expects the memory managers alloc fn to return an af_err, not to
  // throw, if a problem with allocation occurred
  auto allocFn = [](af_memory_manager manager,
                    void** ptr,
                    /* bool */ int userLock,
                    const unsigned ndims,
                    dim_t* dims,
                    const unsigned elSize) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    try {
      *ptr = m->alloc(userLock, ndims, dims, elSize);
    } catch (af::exception& ex) {
      m->log(
          "allocFn: alloc failed with af exception " +
          std::to_string(ex.err()));
      return ex.err(); // AF_ERR_NO_MEM, ...
    } catch (...) {
      m->log("allocFn: alloc failed with unspecified exception");
      return af_err(AF_ERR_UNKNOWN);
    }
    // Log
    m->log(
        "alloc",
        /* size */ dims[0], // HACK: dims[0] until af::memAlloc is size-aware
        userLock,
        (std::uintptr_t)ptr);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_alloc_fn(itf, allocFn));
  auto allocatedFn = [](af_memory_manager manager, size_t* size, void* ptr) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("allocated", (std::uintptr_t)ptr);
    *size = m->allocated(ptr);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_allocated_fn(itf, allocatedFn));
  auto unlockFn = [](af_memory_manager manager, void* ptr, int userLock) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("unlock", (std::uintptr_t)ptr, userLock);
    m->unlock(ptr, (bool)userLock);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_unlock_fn(itf, unlockFn));
  auto signalMemoryCleanupFn = [](af_memory_manager manager) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("signalMemoryCleanup");
    m->signalMemoryCleanup();
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_signal_memory_cleanup_fn(
      itf, signalMemoryCleanupFn));
  auto printInfoFn = [](af_memory_manager manager, char* msg, int device) {
    // no log
    MemoryManagerInstaller::getImpl(manager)->printInfo(msg, device);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_print_info_fn(itf, printInfoFn));
  auto userLockFn = [](af_memory_manager manager, void* ptr) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("userLock", (std::uintptr_t)ptr);
    m->userLock(ptr);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_user_lock_fn(itf, userLockFn));
  auto userUnlockFn = [](af_memory_manager manager, void* ptr) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("userUnlock", (std::uintptr_t)ptr);
    MemoryManagerInstaller::getImpl(manager)->userUnlock(ptr);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_user_unlock_fn(itf, userUnlockFn));
  auto isUserLockedFn = [](af_memory_manager manager, int* out, void* ptr) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("isUserLocked", (std::uintptr_t)ptr);
    *out = (int)m->isUserLocked(ptr);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_is_user_locked_fn(itf, isUserLockedFn));
  auto getMemoryPressureFn = [](af_memory_manager manager, float* pressure) {
    *pressure = MemoryManagerInstaller::getImpl(manager)->getMemoryPressure();
    return AF_SUCCESS;
  };
  AF_CHECK(
      af_memory_manager_set_get_memory_pressure_fn(itf, getMemoryPressureFn));
  auto jitTreeExceedsMemoryPressureFn =
      [](af_memory_manager manager, int* out, size_t bytes) {
        *out = (int)MemoryManagerInstaller::getImpl(manager)
                   ->jitTreeExceedsMemoryPressure(bytes);
        return AF_SUCCESS;
      };
  auto jitTreeExceedsMemoryPressureFn =
      [](af_memory_manager manager, int* out, size_t bytes) {
        *out = (int)MemoryManagerInstaller::getImpl(manager)
                   ->jitTreeExceedsMemoryPressure(bytes);
        return AF_SUCCESS;
      };
  AF_CHECK(af_memory_manager_set_jit_tree_exceeds_memory_pressure_fn(
      itf, jitTreeExceedsMemoryPressureFn));
  auto addMemoryManagementFn = [](af_memory_manager manager, int device) {
    MemoryManagerInstaller::getImpl(manager)->addMemoryManagement(device);
  };
  AF_CHECK(af_memory_manager_set_add_memory_management_fn(
      itf, addMemoryManagementFn));
  auto removeMemoryManagementFn = [](af_memory_manager manager, int device) {
    MemoryManagerInstaller::getImpl(manager)->removeMemoryManagement(device);
  };
  AF_CHECK(af_memory_manager_set_remove_memory_management_fn(
      itf, removeMemoryManagementFn));

  // Native and device memory manager functions
  auto getActiveDeviceIdFn = [itf]() {
    int id;
    AF_CHECK(af_memory_manager_get_active_device_id(itf, &id));
    return id;
  };
  impl_->deviceInterface->getActiveDeviceId = std::move(getActiveDeviceIdFn);
  auto getMaxMemorySizeFn = [itf](int id) {
    size_t out;
    AF_CHECK(af_memory_manager_get_max_memory_size(itf, &out, id));
    return out;
  };
  impl_->deviceInterface->getMaxMemorySize = std::move(getMaxMemorySizeFn);
  // nativeAlloc could throw via AF_CHECK:
  auto nativeAllocFn = [itf](const size_t bytes) {
    void* ptr;
    AF_CHECK(af_memory_manager_native_alloc(itf, &ptr, bytes));
    MemoryManagerInstaller::getImpl(itf)->log(
        "nativeAlloc", bytes, (std::uintptr_t)ptr);
    return ptr;
  };
  impl_->deviceInterface->nativeAlloc = std::move(nativeAllocFn);
  auto nativeFreeFn = [itf](void* ptr) {
    MemoryManagerInstaller::getImpl(itf)->log(
        "nativeFree", (std::uintptr_t)ptr);
    AF_CHECK(af_memory_manager_native_free(itf, ptr));
  };
  impl_->deviceInterface->nativeFree = std::move(nativeFreeFn);
  auto getMemoryPressureThresholdFn = [itf]() {
    float pressure;
    AF_CHECK(af_memory_manager_get_memory_pressure_threshold(itf, &pressure));
    return pressure;
  };
  impl_->deviceInterface->getMemoryPressureThreshold =
      std::move(getMemoryPressureThresholdFn);
  auto setMemoryPressureThresholdFn = [itf](float pressure) {
    AF_CHECK(af_memory_manager_set_memory_pressure_threshold(itf, pressure));
  };
  impl_->deviceInterface->setMemoryPressureThreshold =
      std::move(setMemoryPressureThresholdFn);
}

void MemoryManagerInstaller::setAsMemoryManager() {
  AF_CHECK(af_set_memory_manager(impl_->getHandle()));
  currentlyInstalledMemoryManager_ = impl_;
}

void MemoryManagerInstaller::setAsMemoryManagerPinned() {
  AF_CHECK(af_set_memory_manager_pinned(impl_->getHandle()));
  currentlyInstalledMemoryManager_ = impl_;
}

MemoryManagerAdapter*
MemoryManagerInstaller::currentlyInstalledMemoryManager() {
  return currentlyInstalledMemoryManager_.get();
}

bool MemoryManagerInstaller::installDefaultMemoryManager() {
  std::call_once(startupMemoryInitialize_, []() {
    auto deviceInterface = std::make_shared<MemoryManagerDeviceInterface>();
    auto adapter = std::make_shared<CachingMemoryManager>(
        af::getDeviceCount(), deviceInterface);
    MemoryManagerInstaller::startupMemoryManagerInstaller_ =
        std::make_shared<MemoryManagerInstaller>(adapter);
    MemoryManagerInstaller::startupMemoryManagerInstaller_
        ->setAsMemoryManager();
  });
  return true;
}

void MemoryManagerInstaller::unsetMemoryManager() {
  // Make sure we don't reset the default AF memory manager if it's set
  if (currentlyInstalledMemoryManager_) {
    AF_CHECK(af_unset_memory_manager());
    currentlyInstalledMemoryManager_ = nullptr;
  }
}

} // namespace fl
