/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/memory.h>

#include <memory>
#include <mutex>

#include "flashlight/fl/tensor/backend/af/mem/MemoryManagerAdapter.h"

namespace fl {

/**
 * Manages memory managers and abstracts away parts of the ArrayFire C memory
 * manager API that enables setting active memory managers in ArrayFire.
 *
 * On construction, the installer modifies a given instance of a
 * `MemoryManagerAdapter` and sets needed closures for its underlying
 * `af_memory_manager` handle. Destruction is a noop -- no state change occurs.
 * If the underlying `MemoryManagerAdapter` still exists, it can still be the
 * active ArrayFire memory manager even if its installer has been destroyed.
 */
class MemoryManagerInstaller {
 public:
  /**
   * Creates a new instance using a `MemoryManagerAdapter`. Uses the adapter's
   * underlying `af_memory_manager` handle and performs the following setup:
   * - Sets all function pointers using the Array/Fire C memory management API
   *   on the underlying `af_memory_manager` handle to point to closures which
   *   call the installed `MemoryManagerAdapter`'s instance methods.
   * - Sets the closures on the adapter's `MemoryManagerDeviceInterface` to call
   *   ArrayFire C-API native device memory management functions which
   *   automatically delegate to the proper backend and are use pre-defined
   *   implementations in ArrayFire internals.
   *
   * @param[in] managerImpl a pointer to the `MemoryManagerAdapter` to be
   * installed.
   */
  explicit MemoryManagerInstaller(
      std::shared_ptr<MemoryManagerAdapter> managerImpl);
  ~MemoryManagerInstaller() = default;

  /**
   * Gets the memory manager adapter used in this instance.
   *
   * @return a pointer to some derived type of `MemoryManagerAdapter`
   */
  template <typename T>
  std::shared_ptr<T> getMemoryManager() const {
    return std::dynamic_pointer_cast<T>(impl_);
  }

  /**
   * Sets this `MemoryManagerInstaller`'s `MemoryManagerAdapter` to be the
   * active memory manager in ArrayFire.
   */
  void setAsMemoryManager();

  /**
   * Sets this `MemoryManagerInstaller`'s `MemoryManagerAdapter` to be the
   * active memory manager for pinned memory operations in ArrayFire.
   */
  void setAsMemoryManagerPinned();

  /**
   * Returns an adapter given a handle. Used to construct C++-style callbacks
   * inside lambdas set on the ArrayFire C memory management API.
   */
  static MemoryManagerAdapter* getImpl(af_memory_manager manager);

  /**
   * Returns the currently installed custom memory manager, or null if none is
   * installed.
   */
  static MemoryManagerAdapter* currentlyInstalledMemoryManager();

  /**
   * Initializes and installs the memory manager defaulted to on startup.
   *
   * Uses a `CachingMemoryManager` by default. Only sets the memory manager -
   * doesn't set an AF pinned memory manager.
   */
  static void installDefaultMemoryManager();

  /**
   * Unsets the currently-set custom ArrayFire memory manager. If no custom
   * memory manager is set, results in a noop, since the default memory manager
   * is set, and unsetting it would result in shutdown/destruction.
   */
  static void unsetMemoryManager();

 private:
  // The given memory manager implementation
  std::shared_ptr<MemoryManagerAdapter> impl_;
  // Points to the impl_ of the most recently installed manager.
  static std::shared_ptr<MemoryManagerAdapter> currentlyInstalledMemoryManager_;
};

} // namespace fl
