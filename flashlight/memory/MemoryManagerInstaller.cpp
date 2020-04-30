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

namespace fl {

std::shared_ptr<MemoryManagerAdapter>
    MemoryManagerInstaller::currentlyInstalledMemoryManager_;

MemoryManagerAdapter* MemoryManagerInstaller::getImpl(
    af_memory_manager manager) {
  void* ptr;
  af_memory_manager_get_payload(manager, &ptr);
  return (MemoryManagerAdapter*)ptr;
}

MemoryManagerInstaller::MemoryManagerInstaller(
    std::shared_ptr<MemoryManagerAdapter> managerImpl,
    bool unsetOnDestruction)
    : impl_(managerImpl), unsetOnDestruction_(unsetOnDestruction) {
  if (!impl_) {
    throw std::invalid_argument(
        "MemoryManagerInstaller::MemoryManagerInstaller - "
        "passed MemoryManagerAdapter is null");
  }

  af_memory_manager interface = impl_->getHandle();
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
  AF_CHECK(af_memory_manager_set_initialize_fn(interface, initializeFn));
  auto shutdownFn = [](af_memory_manager manager) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("shutdown");
    m->shutdown();
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_shutdown_fn(interface, shutdownFn));
  auto allocFn = [](af_memory_manager manager,
                    void** ptr,
                    /* bool */ int userLock,
                    const unsigned ndims,
                    dim_t* dims,
                    const unsigned elSize) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    *ptr = m->alloc(userLock, ndims, dims, elSize);
    // Log
    m->log(
        "alloc",
        /* size */ dims[0], // HACK: dims[0] until af::memAlloc is size-aware
        userLock,
        (std::uintptr_t)ptr);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_alloc_fn(interface, allocFn));
  auto allocatedFn = [](af_memory_manager manager, size_t* size, void* ptr) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("allocated", (std::uintptr_t)ptr);
    *size = m->allocated(ptr);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_allocated_fn(interface, allocatedFn));
  auto unlockFn = [](af_memory_manager manager, void* ptr, int userLock) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("unlock", (std::uintptr_t)ptr, userLock);
    m->unlock(ptr, (bool)userLock);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_unlock_fn(interface, unlockFn));
  auto signalMemoryCleanupFn = [](af_memory_manager manager) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("signalMemoryCleanup");
    m->signalMemoryCleanup();
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_signal_memory_cleanup_fn(
      interface, signalMemoryCleanupFn));
  auto printInfoFn = [](af_memory_manager manager, char* msg, int device) {
    // no log
    MemoryManagerInstaller::getImpl(manager)->printInfo(msg, device);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_print_info_fn(interface, printInfoFn));
  auto userLockFn = [](af_memory_manager manager, void* ptr) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("userLock", (std::uintptr_t)ptr);
    m->userLock(ptr);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_user_lock_fn(interface, userLockFn));
  auto userUnlockFn = [](af_memory_manager manager, void* ptr) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("userUnlock", (std::uintptr_t)ptr);
    MemoryManagerInstaller::getImpl(manager)->userUnlock(ptr);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_user_unlock_fn(interface, userUnlockFn));
  auto isUserLockedFn = [](af_memory_manager manager, int* out, void* ptr) {
    MemoryManagerAdapter* m = MemoryManagerInstaller::getImpl(manager);
    m->log("isUserLocked", (std::uintptr_t)ptr);
    *out = (int)m->isUserLocked(ptr);
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_is_user_locked_fn(interface, isUserLockedFn));
  auto getMemoryPressureFn = [](af_memory_manager manager, float* pressure) {
    *pressure = MemoryManagerInstaller::getImpl(manager)->getMemoryPressure();
    return AF_SUCCESS;
  };
  AF_CHECK(af_memory_manager_set_get_memory_pressure_fn(
      interface, getMemoryPressureFn));
  auto jitTreeExceedsMemoryPressureFn =
      [](af_memory_manager manager, int* out, size_t bytes) {
        *out = (int)MemoryManagerInstaller::getImpl(manager)
                   ->jitTreeExceedsMemoryPressure(bytes);
        return AF_SUCCESS;
      };
  AF_CHECK(af_memory_manager_set_jit_tree_exceeds_memory_pressure_fn(
      interface, jitTreeExceedsMemoryPressureFn));
  auto addMemoryManagementFn = [](af_memory_manager manager, int device) {
    MemoryManagerInstaller::getImpl(manager)->addMemoryManagement(device);
  };
  AF_CHECK(af_memory_manager_set_add_memory_management_fn(
      interface, addMemoryManagementFn));
  auto removeMemoryManagementFn = [](af_memory_manager manager, int device) {
    MemoryManagerInstaller::getImpl(manager)->removeMemoryManagement(device);
  };
  AF_CHECK(af_memory_manager_set_remove_memory_management_fn(
      interface, removeMemoryManagementFn));

  // Native and device memory manager functions
  auto getActiveDeviceIdFn = [interface]() {
    int id;
    AF_CHECK(af_memory_manager_get_active_device_id(interface, &id));
    return id;
  };
  impl_->deviceInterface->getActiveDeviceId = std::move(getActiveDeviceIdFn);
  auto getMaxMemorySizeFn = [interface](int id) {
    size_t out;
    AF_CHECK(af_memory_manager_get_max_memory_size(interface, &out, id));
    return out;
  };
  impl_->deviceInterface->getMaxMemorySize = std::move(getMaxMemorySizeFn);
  auto nativeAllocFn = [interface](const size_t bytes) {
    void* ptr;
    AF_CHECK(af_memory_manager_native_alloc(interface, &ptr, bytes));
    MemoryManagerInstaller::getImpl(interface)->log(
        "nativeAlloc", bytes, (std::uintptr_t)ptr);
    return ptr;
  };
  impl_->deviceInterface->nativeAlloc = std::move(nativeAllocFn);
  auto nativeFreeFn = [interface](void* ptr) {
    MemoryManagerInstaller::getImpl(interface)->log(
        "nativeFree", (std::uintptr_t)ptr);
    AF_CHECK(af_memory_manager_native_free(interface, ptr));
  };
  impl_->deviceInterface->nativeFree = std::move(nativeFreeFn);
  auto getMemoryPressureThresholdFn = [interface]() {
    float pressure;
    AF_CHECK(
        af_memory_manager_get_memory_pressure_threshold(interface, &pressure));
    return pressure;
  };
  impl_->deviceInterface->getMemoryPressureThreshold =
      std::move(getMemoryPressureThresholdFn);
  auto setMemoryPressureThresholdFn = [interface](float pressure) {
    AF_CHECK(
        af_memory_manager_set_memory_pressure_threshold(interface, pressure));
  };
  impl_->deviceInterface->setMemoryPressureThreshold =
      std::move(setMemoryPressureThresholdFn);
}

MemoryManagerInstaller::~MemoryManagerInstaller() {
  if (unsetOnDestruction_) {
    try {
      AF_CHECK(af_unset_memory_manager());
      AF_CHECK(af_unset_memory_manager_pinned());
    } catch (std::exception& e) {
      LOG(ERROR) << "MemoryManagerInstaller::~MemoryManagerInstaller() "
                    "failed to unset ArrayFire memory manager with error="
                 << e.what();
      std::exit(-1);
    }
    currentlyInstalledMemoryManager_ = nullptr;
  }
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

size_t afGetMemStepSize() {
  MemoryManagerAdapter* customMemoryManager =
      MemoryManagerInstaller::currentlyInstalledMemoryManager();
  if (customMemoryManager) {
    return customMemoryManager->getMemStepSize();
  } else {
    return af::getMemStepSize();
  }
}

void afSetMemStepSize(size_t size) {
  MemoryManagerAdapter* customMemoryManager =
      MemoryManagerInstaller::currentlyInstalledMemoryManager();
  if (customMemoryManager) {
    customMemoryManager->setMemStepSize(size);
  } else {
    af::setMemStepSize(size);
  }
}

} // namespace fl
