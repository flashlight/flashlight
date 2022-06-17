/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
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
 * Return a readable string representation of the given device type.
 *
 * @return a string that represents the given device type.
 */
std::string deviceTypeToString(const DeviceType type);

/**
 * Output a string representation of `type` to `os`.
 */
std::ostream& operator<<(std::ostream& os, const DeviceType& type);

/**
 * Returns all device types.
 *
 * @return an immutable reference to a set of all device types.
 */
const std::unordered_set<DeviceType>& getDeviceTypes();

} // namespace fl
