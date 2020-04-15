/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/common/DevicePtr.h"

using namespace fl;

namespace {

bool isLockedArray(const af::array& a) {
  bool res;
  auto err = af_is_locked_array(&res, a.get());
  if (err != AF_SUCCESS) {
    throw std::runtime_error(
        "af_is_locked_array returned error: " + std::to_string(err));
  }
  return res;
}

} // namespace

TEST(DevicePtrTest, Null) {
  af::array x;
  DevicePtr xp(x);
  EXPECT_EQ(xp.get(), nullptr);
}

TEST(DevicePtrTest, Locking) {
  af::array x(3, 3);
  EXPECT_FALSE(isLockedArray(x));
  {
    DevicePtr xp(x);
    EXPECT_TRUE(isLockedArray(x));
  }
  EXPECT_FALSE(isLockedArray(x));
}

TEST(DevicePtrTest, Move) {
  af::array x(3, 3);
  af::array y(4, 4);

  DevicePtr yp(y);
  EXPECT_FALSE(isLockedArray(x));
  EXPECT_TRUE(isLockedArray(y));

  yp = DevicePtr(x);
  EXPECT_TRUE(isLockedArray(x));
  EXPECT_FALSE(isLockedArray(y));

  yp = DevicePtr();
  EXPECT_FALSE(isLockedArray(x));
  EXPECT_FALSE(isLockedArray(y));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
