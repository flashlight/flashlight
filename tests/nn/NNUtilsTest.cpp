/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/autograd/autograd.h"
#include "flashlight/nn/nn.h"

using namespace fl;

TEST(UtilsTest, Join) {
  // Empty vector
  auto empty = join({});
  ASSERT_TRUE(empty.isempty());

  // Single array
  auto i = af::randu(50, 60, 70);
  auto o = join({i}, -1, 3);
  ASSERT_TRUE(af::allTrue<bool>(o == i));

  // more than one arrays
  auto a = af::constant(1, 25, 1, 300);
  auto b = af::constant(2, 20, 1, 300);
  auto c = af::constant(3, 30, 1, 300);

  auto o1 = join({a, b, c});
  ASSERT_EQ(o1.dims(), af::dim4(30, 1, 300, 3));
  ASSERT_TRUE(af::allTrue<bool>(o1(af::seq(25), 0, af::seq(300), 0) == a));
  ASSERT_TRUE(af::allTrue<bool>(o1(af::seq(25, 29), 0, af::seq(300), 0) == 0));
  ASSERT_TRUE(af::allTrue<bool>(o1(af::seq(20), 0, af::seq(300), 1) == b));
  ASSERT_TRUE(af::allTrue<bool>(o1(af::seq(20, 29), 0, af::seq(300), 1) == 0));
  ASSERT_TRUE(af::allTrue<bool>(o1(af::seq(30), 0, af::seq(300), 2) == c));

  auto o2 = join({a, b, c}, -1);
  ASSERT_EQ(o2.dims(), af::dim4(30, 1, 300, 3));
  ASSERT_TRUE(af::allTrue<bool>(o2(af::seq(25), 0, af::seq(300), 0) == a));
  ASSERT_TRUE(af::allTrue<bool>(o2(af::seq(25, 29), 0, af::seq(300), 0) == -1));
  ASSERT_TRUE(af::allTrue<bool>(o2(af::seq(20), 0, af::seq(300), 1) == b));
  ASSERT_TRUE(af::allTrue<bool>(o2(af::seq(20, 29), 0, af::seq(300), 1) == -1));
  ASSERT_TRUE(af::allTrue<bool>(o2(af::seq(30), 0, af::seq(300), 2) == c));

  auto o3 = join({a, b, c}, -1, 1);
  ASSERT_EQ(o3.dims(), af::dim4(30, 3, 300));
  ASSERT_TRUE(af::allTrue<bool>(o3(af::seq(25), 0, af::seq(300)) == a));
  ASSERT_TRUE(af::allTrue<bool>(o3(af::seq(25, 29), 0, af::seq(300)) == -1));
  ASSERT_TRUE(af::allTrue<bool>(o3(af::seq(20), 1, af::seq(300)) == b));
  ASSERT_TRUE(af::allTrue<bool>(o3(af::seq(20, 29), 1, af::seq(300)) == -1));
  ASSERT_TRUE(af::allTrue<bool>(o3(af::seq(30), 2, af::seq(300)) == c));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
