/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <stdexcept>
#include <unordered_map>

#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/tensor/Init.h"

using fl::DeviceManager;
using fl::DeviceType;

TEST(DeviceManagerTest, getInstance) {
  ASSERT_EQ(&DeviceManager::getInstance(), &DeviceManager::getInstance());
}

TEST(DeviceManagerTest, isDeviceTypeAvailable) {
  auto& manager = DeviceManager::getInstance();
  // x64 (CPU) should be always available
  ASSERT_TRUE(manager.isDeviceTypeAvailable(DeviceType::x64));

  // CUDA availability depends on compilation
  bool expectCUDA = FL_BACKEND_CUDA;
  ASSERT_EQ(manager.isDeviceTypeAvailable(DeviceType::CUDA), expectCUDA);
}

TEST(DeviceManagerTest, getDeviceCount) {
  auto& manager = DeviceManager::getInstance();
  // For now we always treat CPU as a single device
  ASSERT_EQ(manager.getDeviceCount(DeviceType::x64), 1);

  if (manager.isDeviceTypeAvailable(DeviceType::CUDA)) {
    ASSERT_NO_THROW(manager.getDeviceCount(DeviceType::CUDA));
  } else {
    ASSERT_THROW(manager.getDeviceCount(DeviceType::CUDA),
      std::invalid_argument);
  }
}

TEST(DeviceManagerTest, getDevicesOfType) {
  auto& manager = DeviceManager::getInstance();
  // For now we always treat CPU as a single device
  ASSERT_EQ(manager.getDevicesOfType(DeviceType::x64).size(), 1);

  for (auto type : fl::getDeviceTypes()) {
    if (manager.isDeviceTypeAvailable(DeviceType::CUDA)) {
      for (auto device : manager.getDevicesOfType(type)) {
        ASSERT_EQ(device->type(), type);
      }
    } else {
      ASSERT_THROW(manager.getDeviceCount(DeviceType::CUDA),
          std::invalid_argument);
    }
  }
}

TEST(DeviceManagerTest, getDevice) {
  auto& manager = DeviceManager::getInstance();
  auto& x64Device =
    manager.getDevice(DeviceType::x64, fl::kX64DeviceId);
  ASSERT_EQ(x64Device.type(), DeviceType::x64);
}

TEST(DeviceManagerTest, getActiveDevice) {
  auto& manager = DeviceManager::getInstance();
  for (auto type : fl::getDeviceTypes()) {
    if (manager.isDeviceTypeAvailable(type)) {
      ASSERT_EQ(manager.getActiveDevice(type).type(), type);
    } else {
      ASSERT_THROW(manager.getActiveDevice(type), std::invalid_argument);
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
