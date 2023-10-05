/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/runtime/Device.h"
#include "flashlight/fl/runtime/DeviceManager.h"

#if FL_BACKEND_CUDA
#include "flashlight/fl/runtime/CUDAUtils.h"
#endif

namespace {

int getActiveDeviceId(const fl::DeviceType type) {
  switch (type) {
    case fl::DeviceType::x64: return fl::kX64DeviceId;
    case fl::DeviceType::CUDA: {
#if FL_BACKEND_CUDA
      return fl::cuda::getActiveDeviceId();
#endif
      throw std::runtime_error("CUDA is unsupported");
    }
  }
}

} // namespace

namespace fl {

DeviceManager::DeviceManager() {
  // initialize for x64
  DeviceTypeInfo x64Info;
  x64Info.emplace(kX64DeviceId, std::make_unique<X64Device>());
  deviceTypeToInfo_.emplace(DeviceType::x64, std::move(x64Info));

  // initialize for CUDA
#if FL_BACKEND_CUDA
  deviceTypeToInfo_.insert({DeviceType::CUDA, fl::cuda::createCUDADevices()});
#endif
}

void DeviceManager::enforceDeviceTypeAvailable(
  std::string_view errorPrefix, const DeviceType type) const {
  if (!isDeviceTypeAvailable(type)) {
    throw std::runtime_error(
      std::string(errorPrefix) + " device type unavailable");
  }
}

DeviceManager& DeviceManager::getInstance() {
  static DeviceManager instance;
  return instance;
}

bool DeviceManager::isDeviceTypeAvailable(const DeviceType type) const {
  return deviceTypeToInfo_.count(type) != 0;
}

unsigned DeviceManager::getDeviceCount(const DeviceType type) const {
  enforceDeviceTypeAvailable("[DeviceManager::getDeviceCount]", type);
  return deviceTypeToInfo_.at(type).size();
}

std::vector<Device*> DeviceManager::getDevicesOfType(
  DeviceType type) {
  enforceDeviceTypeAvailable("[DeviceManager::getDevicesOfType]", type);
  std::vector<Device*> devices;
  for (auto &[_, device] : deviceTypeToInfo_.at(type)) {
    devices.push_back(device.get());
  }
  return devices;
}

std::vector<const Device*> DeviceManager::getDevicesOfType(
  DeviceType type) const {
  enforceDeviceTypeAvailable("[DeviceManager::getDevicesOfType]", type);
  std::vector<const Device*> devices;
  for (auto &[_, device] : deviceTypeToInfo_.at(type)) {
    devices.push_back(device.get());
  }
  return devices;
}

Device& DeviceManager::getDevice(const DeviceType type, int id) const {
  enforceDeviceTypeAvailable("[DeviceManager::getActiveDevice]", type);
  auto& idToDevice = deviceTypeToInfo_.at(type);
  if (idToDevice.count(id) == 0) {
    throw std::runtime_error(
      "[DeviceManager::getDevice] unknown device id");
  }
  return *idToDevice.at(id);
}

Device& DeviceManager::getActiveDevice(const DeviceType type) const {
  enforceDeviceTypeAvailable("[DeviceManager::getActiveDevice]", type);
  int activeDeviceId = getActiveDeviceId(type);
  return *deviceTypeToInfo_.at(type).at(activeDeviceId);
}

} // namespace fl
