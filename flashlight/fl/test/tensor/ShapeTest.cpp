/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arrayfire.h>
#include <gtest/gtest.h>

#include <stdexcept>

#include "flashlight/fl/tensor/Shape.h"

// TODO: move me to an independent AF test suite
#include "flashlight/fl/tensor/backend/af/Utils.h"

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
  ASSERT_EQ(Shape({1, 1, 1}).nDims(), 3);
  ASSERT_EQ(Shape({5, 2, 3}).nDims(), 3);
  ASSERT_EQ(Shape({1, 2, 3, 6}).nDims(), 4);
  if (Shape::kMaxDims > 4) {
    ASSERT_EQ(Shape({1, 2, 3, 1, 1, 1}).nDims(), 6);
    ASSERT_EQ(Shape({1, 2, 3, 1, 1, 1, 5}).nDims(), 7);
    ASSERT_EQ(Shape({4, 2, 3, 1, 1, 1, 5}).nDims(), 7);
  }
}

TEST(ShapeTest, elements) {
  ASSERT_EQ(Shape().elements(), 0);
  ASSERT_EQ(Shape({1, 1, 1, 1}).elements(), 1);
  ASSERT_EQ(Shape({1, 2, 3, 4}).elements(), 24);
  ASSERT_EQ(Shape({1, 2, 3, 0}).elements(), 0);
}

TEST(ShapeTest, Equality) {
  auto a = Shape({1, 2, 3, 4});
  ASSERT_EQ(a, Shape({1, 2, 3, 4}));
  ASSERT_NE(a, Shape({4, 3, 4}));
  ASSERT_NE(Shape({1, 2}), Shape({1, 1, 1, 2}));
  ASSERT_NE(Shape({5, 2, 3}), Shape({5, 2, 3, 1}));
  ASSERT_EQ(Shape({5, 2, 3, 1}), Shape({5, 2, 3, 1}));
  ASSERT_NE(Shape({5, 2, 1, 1}), Shape({5, 2, 1, 4}));
}

// TODO: move me to an independent AF test suite
TEST(ShapeTest, ArrayFireInterop) {
  auto dimsEq = [](const af::dim4& d, const Shape& s) {
    return detail::afToFlDims(d) == s;
  };

  ASSERT_TRUE(dimsEq(af::dim4(), {}));
  ASSERT_TRUE(dimsEq(af::dim4(3), {3})); // not 3, 1, 1, 1
  ASSERT_TRUE(dimsEq(af::dim4(3, 2), {3, 2})); // not 3, 2, 1, 1
  ASSERT_TRUE(dimsEq(af::dim4(3, 1), {3})); // 1 is indistinguishable in AF
  ASSERT_TRUE(dimsEq(af::dim4(1, 3, 2), {1, 3, 2}));
  ASSERT_TRUE(dimsEq(af::dim4(1), {1}));
  ASSERT_TRUE(dimsEq(af::dim4(1, 1, 1), {1}));
}
