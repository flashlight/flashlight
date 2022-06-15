/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/Device.h"
#include <unordered_set>

namespace fl {

const std::unordered_set<DeviceType>& getDeviceTypes() {
  static std::unordered_set<DeviceType> types = {
    DeviceType::x64,
    DeviceType::CUDA
  };
  return types;
}

} // namespace fl
