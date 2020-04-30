
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <mutex>

#include <arrayfire.h>

#include "flashlight/common/CppBackports.h"
#include "flashlight/memory/memory.h"

namespace fl {

/**
 * Initializes and installs the default memory manager.
 *
 * Uses a CachingMemoryManager by default.
 */
int initializeDefaultMemoryManager() {
  // TODO: is this needed?
  // af::init(); // ensure the AF device manager is initialized

  static std::once_flag memInitialize;
  // Needs to stay in scope else the manager will be unset on destruction
  std::call_once(memInitialize, []() {
    auto deviceInterface = std::make_shared<MemoryManagerDeviceInterface>();
    auto adapter = std::make_shared<CachingMemoryManager>(
        af::getDeviceCount(), deviceInterface);

    // This installer falls out of scope but doesn't unset the existing memory
    // manager since it should persist
    auto defaultMemoryManagerInstaller =
        std::make_shared<MemoryManagerInstaller>(
            adapter, /*unsetOnDestruction=*/false);
    defaultMemoryManagerInstaller->setAsMemoryManager();
  });
  return 0;
}

namespace {

int ret = initializeDefaultMemoryManager();

}

} // namespace fl
