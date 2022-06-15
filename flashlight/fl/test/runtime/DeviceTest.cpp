/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/runtime/Device.h"
#include "flashlight/fl/tensor/Init.h"

using fl::DeviceType;

TEST(DeviceTest, getAllDeviceTypes) {
  auto allDevices = fl::getDeviceTypes();
  ASSERT_TRUE(allDevices.count(DeviceType::x64) == 1);
  ASSERT_TRUE(allDevices.count(DeviceType::CUDA) == 1);
  ASSERT_EQ(allDevices.size(), 2);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
