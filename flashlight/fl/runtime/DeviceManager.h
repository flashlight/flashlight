/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/runtime/Device.h"

namespace fl {

/**
 * Device id for the single CPU device.
 */
constexpr int kX64DeviceId = 0;

/**
 * A singleton to manage all supported types of devices.
 */
class FL_API DeviceManager {
  using DeviceTypeInfo = std::unordered_map<int, const std::unique_ptr<Device>>;

  std::unordered_map<DeviceType, DeviceTypeInfo> deviceTypeToInfo_;

  // Help enforce singleton
  DeviceManager();
  DeviceManager(const DeviceManager&) = delete;
  DeviceManager(DeviceManager&&) = delete;
  DeviceManager& operator=(const DeviceManager&) = delete;
  DeviceManager& operator=(DeviceManager&&) = delete;

  // throws runtime_error if `type` is unavailable
  void enforceDeviceTypeAvailable(
      std::string_view errorPrefix,
      const DeviceType type) const;

 public:
  /**
   * Gets the singleton DeviceManager.
   *
   * @return a reference to the singleton DeviceManager.
   */
  static DeviceManager& getInstance();

  /**
   * Returns if the given device type is available.
   *
   * @return a boolean denoting device type availability.
   */
  bool isDeviceTypeAvailable(const DeviceType type) const;

  /**
   * Gets the number of usable devices of given type.
   *
   * Throws a runtime_error if given device `type` is unavailable.
   *
   * @return the number of usable devices of given type.
   */
  unsigned getDeviceCount(const DeviceType type) const;

  /**
   * Gets all devices of given type.
   *
   * Throws a runtime_error if given device `type` is unavailable.
   *
   * @return a vector of pointers to all devices of given type.
   */
  std::vector<Device*> getDevicesOfType(const DeviceType type);

  /**
   * Gets all devices of given type.
   *
   * Throws a runtime_error if given device `type` is unavailable.
   *
   * @return a vector of immutable pointers to all devices of given type.
   */
  std::vector<const Device*> getDevicesOfType(const DeviceType type) const;

  /**
   * Gets the device of given type and native device id.
   *
   * Throws a runtime_error if given device `type` is unavailable
   * or `id` does not match any device.
   *
   * @return a reference to the device of given type and native device id.
   */
  Device& getDevice(const DeviceType type, int id) const;

  /**
   * Gets the active device of given type.
   *
   * Throws a runtime_error if given device `type` is unavailable.
   *
   * @return a reference to the active device of given type.
   */
  Device& getActiveDevice(const DeviceType type) const;
};

} // namespace fl
