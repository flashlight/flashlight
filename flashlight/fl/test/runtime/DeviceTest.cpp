/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/tensor/Init.h"

using fl::DeviceManager;

TEST(DeviceTest, type) {
  auto& manager = DeviceManager::getInstance();
  for (auto type : fl::getDeviceTypes()) {
    if (manager.isDeviceTypeAvailable(type)) {
      for (auto* device : manager.getDevicesOfType(type)) {
        ASSERT_EQ(device->type(), type);
      }
    }
  }
}

TEST(DeviceTest, setActive) {
  auto& manager = DeviceManager::getInstance();
  for (auto type : fl::getDeviceTypes()) {
    if (manager.isDeviceTypeAvailable(type)) {
      for (auto* device : manager.getDevicesOfType(type)) {
        device->setActive();
        ASSERT_EQ(&manager.getActiveDevice(type), device);
      }
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
