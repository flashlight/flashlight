/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_set>

namespace fl {

/**
 * A runtime type for various device types.
 * NOTE update `fl::getAllDeviceTypes` after changing enum values.
 */
enum class DeviceType {
  x64,
  CUDA,
};

/**
 * Returns all device types.
 *
 * @return an immutable reference to a set of all device types.
 */
const std::unordered_set<DeviceType>& getDeviceTypes();

/**
 * A Device abstraction.
 */
struct Device {
  DeviceType type;
  explicit Device(DeviceType type) : type(type) {}

  // no copy/move
  Device(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(const Device&) = delete;
  Device& operator=(Device&&) = delete;

  // TODO more functionalities
};

} // namespace fl
