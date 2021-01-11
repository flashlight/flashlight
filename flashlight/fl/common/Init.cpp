/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mutex>

#include <af/device.h>

#include "flashlight/fl/memory/MemoryManagerInstaller.h"

namespace fl {
namespace {
std::once_flag flInitFlag;
}

/**
 * Initialize Flashlight. Performs setup, including:
 * - Ensures ArrayFire globals are initialized
 * - Sets the default memory manager (CachingMemoryManager)
 *
 * Can only be called once per process. Subsequent calls will be noops.
 */
void init() {
  std::call_once(flInitFlag, []() {
    af_init();
    MemoryManagerInstaller::installDefaultMemoryManager();
  });
}

} // namespace fl
