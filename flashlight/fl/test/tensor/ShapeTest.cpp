/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arrayfire.h>
#include <gtest/gtest.h>

#include <stdexcept>

#include "flashlight/fl/tensor/ShapeBase.h"

using namespace ::testing;
using namespace fl;

TEST(ShapeTest, Basic) {
  auto s = Shape({3, 4});
  ASSERT_EQ(s.nDims(), 2);
  ASSERT_EQ(s.dim(0), 3);
  ASSERT_EQ(s.dim(1), 4);
  EXPECT_THROW(s.dim(5), std::invalid_argument);
}

TEST(ShapeTest, ManyDims) {
  if (Shape::kMaxDims <= 4) {
    GTEST_SKIP() << "Max shape dimensions is <= 4";
  }
  auto many = Shape({1, 2, 3, 4, 5, 6, 7});
  ASSERT_EQ(many.nDims(), 7);
  ASSERT_EQ(many.dim(5), 6);
}

TEST(ShapeTest, nDims) {
  ASSERT_EQ(Shape().nDims(), 0);
  ASSERT_EQ(Shape({1, 1, 1}).nDims(), 1);
  ASSERT_EQ(Shape({5, 2, 3}).nDims(), 3);
  ASSERT_EQ(Shape({1, 2, 3, 6}).nDims(), 4); // leading 1
  std::cout << "Shape max dims " << Shape::kMaxDims << std::endl;
  if (Shape::kMaxDims > 4) {
    ASSERT_EQ(Shape({1, 2, 3, 1, 1, 1}).nDims(), 3);
    ASSERT_EQ(Shape({1, 2, 3, 1, 1, 1, 5}).nDims(), 7);
    ASSERT_EQ(Shape({4, 2, 3, 1, 1, 1, 5}).nDims(), 7);
  }
}

TEST(ShapeTest, Equality) {
  // TODO: test fixtures for every type, etc.
  auto a = Shape({1, 2, 3, 4});
  ASSERT_EQ(a, Shape({1, 2, 3, 4}));
  ASSERT_NE(a, Shape({4, 3, 4}));
  ASSERT_NE(Shape({1, 2}), Shape({1, 1, 1, 2}));
  ASSERT_EQ(Shape({5, 2, 3}), Shape({5, 2, 3, 1}));
  ASSERT_NE(Shape({5, 2, 1, 1}), Shape({5, 2, 1, 4}));
}
