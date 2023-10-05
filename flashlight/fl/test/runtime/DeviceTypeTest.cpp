/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/runtime/DeviceType.h"

using fl::DeviceType;

TEST(DeviceTypeTest, getAllDeviceTypes) {
  const auto& allDevices = fl::getDeviceTypes();
  ASSERT_TRUE(allDevices.count(DeviceType::x64) == 1);
  ASSERT_TRUE(allDevices.count(DeviceType::CUDA) == 1);
  ASSERT_EQ(allDevices.size(), 2);
}

TEST(DeviceTypeTest, deviceTypeToString) {
  ASSERT_EQ(deviceTypeToString(DeviceType::x64), "x64");
  ASSERT_EQ(deviceTypeToString(DeviceType::CUDA), "CUDA");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
