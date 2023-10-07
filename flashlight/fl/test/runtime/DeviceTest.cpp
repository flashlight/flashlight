/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/tensor/Init.h"

using fl::DeviceManager;
using fl::DeviceType;

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

TEST(DeviceTest, nativeId) {
  const auto& manager = DeviceManager::getInstance();
  for (const auto* device : manager.getDevicesOfType(DeviceType::x64)) {
    ASSERT_EQ(device->nativeId(), fl::kX64DeviceId);
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

TEST(DeviceTest, addSetActiveCallback) {
  auto& manager = DeviceManager::getInstance();
  for (const auto type : fl::getDeviceTypes()) {
    if (manager.isDeviceTypeAvailable(type)) {
      for (auto* device : manager.getDevicesOfType(type)) {
        int count = 0;
        auto incCount = [&count](int){ count++; };
        device->addSetActiveCallback(incCount);
        device->setActive();
        ASSERT_EQ(count, 1);
      }
    }
  }
}

TEST(DeviceTest, sync) {
  const auto& manager = DeviceManager::getInstance();
  for (const auto type : fl::getDeviceTypes()) {
    if (manager.isDeviceTypeAvailable(type)) {
      for (const auto* device : manager.getDevicesOfType(type)) {
        ASSERT_NO_THROW(device->sync());
      }
    }
  }
}

TEST(DeviceTest, getStream) {
  auto& manager = DeviceManager::getInstance();
  for (const auto type : fl::getDeviceTypes()) {
    if (manager.isDeviceTypeAvailable(type)) {
      for (const auto* device : manager.getDevicesOfType(type)) {
        for (const auto& stream : device->getStreams()) {
          ASSERT_EQ(&stream->device(), device);
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
