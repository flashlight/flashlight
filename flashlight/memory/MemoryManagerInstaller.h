/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/memory.h>

#include <memory>

#include "flashlight/memory/MemoryManagerAdapter.h"

namespace fl {

/**
 * A base class that manages memory managers and abstracts away parts of the
 * ArrayFire C memory manager API so stateful C++ classes can be used to
 * implement memory managers.
 */
class MemoryManagerInstaller {
 public:
  ~MemoryManagerInstaller() = default;
  MemoryManagerInstaller(std::shared_ptr<MemoryManagerAdapter> managerImpl);

  template <typename T>
  std::shared_ptr<T> getMemoryManager() const {
    return std::dynamic_pointer_cast<T>(impl_);
  }

  void setAsMemoryManager();
  void setAsMemoryManagerPinned();

  static MemoryManagerAdapter* getImpl(af_memory_manager manager);

 private:
  // The given memory manager implementation
  std::shared_ptr<MemoryManagerAdapter> impl_;
};

} // namespace fl
