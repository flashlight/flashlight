/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mutex>

#include <af/device.h>

#include "flashlight/fl/memory/MemoryManagerInstaller.h"
#include "flashlight/fl/memory/managers/CachingMemoryManager.h"

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
void init(int memRecyclingSize /*=-1*/, int memSplitSize /*=-1*/) {
  std::call_once(flInitFlag, [memRecyclingSize, memSplitSize]() {
    af_init();
    // TODO: remove this temporary workaround for TextDatasetTest crash on CPU
    // backend when tearing down the test environment. This is possibly due to
    // AF race conditions when tearing down our custom memory manager.
    if (!FL_BACKEND_CPU) {
      MemoryManagerInstaller::installDefaultMemoryManager();
      auto* curMemMgr =
          fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
      if (curMemMgr) {
        auto cachMemMgr = dynamic_cast<fl::CachingMemoryManager*>(curMemMgr);
        if (cachMemMgr) {
          if (memRecyclingSize > -1) {
            cachMemMgr->setRecyclingSizeLimit(memRecyclingSize);
          }
          if (memSplitSize > -1) {
            cachMemMgr->setSplitSizeLimit(memSplitSize);
          }
        }
      }
    }
  });
}

} // namespace fl
