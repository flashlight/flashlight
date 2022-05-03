/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mutex>

#include <iostream>
#include <string>

#include "flashlight/fl/tensor/TensorBackend.h"

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
  std::call_once(flInitFlag, []() { Tensor().backend(); });
}

} // namespace fl
