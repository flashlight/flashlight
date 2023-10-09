/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/TensorBase.h"

using namespace fl;

TEST(DevicePtrTest, Null) {
  Tensor x;
  DevicePtr xp(x);
  EXPECT_EQ(xp.get(), nullptr);
}

TEST(DevicePtrTest, NoCopy) {
  Tensor a = fl::full({3, 3}, 5.);

  void* devicePtrLoc;
  {
    DevicePtr p(a);
    devicePtrLoc = p.get();
  }
  EXPECT_EQ(devicePtrLoc, a.device<void>());
  a.unlock();
}

TEST(DevicePtrTest, Locking) {
  Tensor x({3, 3});
  EXPECT_FALSE(x.isLocked());
  {
    DevicePtr xp(x);
    EXPECT_TRUE(x.isLocked());
  }
  EXPECT_FALSE(x.isLocked());
}

TEST(DevicePtrTest, Move) {
  Tensor x({3, 3});
  Tensor y({4, 4});

  DevicePtr yp(y);
  EXPECT_FALSE(x.isLocked());
  EXPECT_TRUE(y.isLocked());

  yp = DevicePtr(x);
  EXPECT_TRUE(x.isLocked());
  EXPECT_FALSE(y.isLocked());

  yp = DevicePtr();
  EXPECT_FALSE(x.isLocked());
  EXPECT_FALSE(y.isLocked());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
