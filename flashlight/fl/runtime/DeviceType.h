/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ostream>
#include <string>
#include <unordered_set>

#include "flashlight/fl/common/Defines.h"

namespace fl {

/**
 * A runtime type for various device types.
 * NOTE update `fl::getAllDeviceTypes` after changing enum values.
 */
enum class DeviceType {
  x64,
  CUDA,
};

#if FL_BACKEND_CUDA
constexpr DeviceType kDefaultDeviceType = DeviceType::CUDA;
#else
constexpr DeviceType kDefaultDeviceType = DeviceType::x64;
#endif

/**
 * Return a readable string representation of the given device type.
 *
 * @return a string that represents the given device type.
 */
FL_API std::string deviceTypeToString(const DeviceType type);

/**
 * Output a string representation of `type` to `os`.
 */
FL_API std::ostream& operator<<(std::ostream& os, const DeviceType& type);

/**
 * Returns all device types.
 *
 * @return an immutable reference to a set of all device types.
 */
FL_API const std::unordered_set<DeviceType>& getDeviceTypes();

} // namespace fl
