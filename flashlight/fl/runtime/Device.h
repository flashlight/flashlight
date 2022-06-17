/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/runtime/DeviceType.h"

namespace fl {

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
