/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Init.h"

#include <mutex>

#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/TensorBackend.h"

namespace fl {
namespace {
std::once_flag flInitFlag;
}

/**
 * Initialize Flashlight. Performs setup, including:
 * - Ensures default tensor backend globals are initialized, including memory
 *   management, tensor backend state, computation stremas, etc.
 * - Sets signal handlers helpful for debugging, if enabled.
 *
 * Body is only run once per process. Subsequent calls will be noops.
 */
void init() {
  std::call_once(flInitFlag, []() {
    defaultTensorBackend();
    initLogging();
  });
}

} // namespace fl
