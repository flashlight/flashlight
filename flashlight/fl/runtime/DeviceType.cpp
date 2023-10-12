/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/DeviceType.h"

namespace fl {

std::string deviceTypeToString(const DeviceType type) {
  switch (type) {
    case DeviceType::x64: return "x64";
    case DeviceType::CUDA: return "CUDA";
  }
}

std::ostream& operator<<(std::ostream& os, const DeviceType& type) {
  return os << deviceTypeToString(type);
}

const std::unordered_set<DeviceType>& getDeviceTypes() {
  static std::unordered_set<DeviceType> types = {
    DeviceType::x64,
    DeviceType::CUDA
  };
  return types;
}

} // namespace fl
