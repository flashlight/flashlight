/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <stdexcept>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Shape.h"

using namespace ::testing;
using namespace fl;

TEST(ShapeTest, Basic) {
  auto s = Shape({3, 4});
  ASSERT_EQ(s.ndim(), 2);
  ASSERT_EQ(s.dim(0), 3);
  ASSERT_EQ(s.dim(1), 4);
  EXPECT_THROW(s.dim(5), std::invalid_argument);
}

TEST(ShapeTest, ManyDims) {
  if (Shape::kMaxDims <= 4) {
    GTEST_SKIP() << "Max shape dimensions is <= 4";
  }
  auto many = Shape({1, 2, 3, 4, 5, 6, 7});
  ASSERT_EQ(many.ndim(), 7);
  ASSERT_EQ(many.dim(5), 6);
}

TEST(ShapeTest, ndim) {
  ASSERT_EQ(Shape().ndim(), 0);
  ASSERT_EQ(Shape({1, 0, 1}).ndim(), 3);
  ASSERT_EQ(Shape({1, 1, 1}).ndim(), 3);
  ASSERT_EQ(Shape({5, 2, 3}).ndim(), 3);
  ASSERT_EQ(Shape({1, 2, 3, 6}).ndim(), 4);
  if (Shape::kMaxDims > 4) {
    ASSERT_EQ(Shape({1, 2, 3, 1, 1, 1}).ndim(), 6);
    ASSERT_EQ(Shape({1, 2, 3, 1, 1, 1, 5}).ndim(), 7);
    ASSERT_EQ(Shape({4, 2, 3, 1, 1, 1, 5}).ndim(), 7);
  }
}

TEST(ShapeTest, elements) {
  ASSERT_EQ(Shape().elements(), 1); // empty shape = scalar
  ASSERT_EQ(Shape({0}).elements(), 0); // empty tensor
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

TEST(ShapeTest, Indexing) {
  auto a = Shape({3, 4, 5, 2});
  ASSERT_EQ(a[0], 3);
  ASSERT_EQ(a[1], 4);
  ASSERT_EQ(a[2], 5);
  ASSERT_EQ(a[3], 2);
  ASSERT_THROW(a[4], std::invalid_argument);
}

TEST(ShapeTest, string) {
  auto checkShapeStrEqual = [](const Shape& s, const std::string& str) -> void {
    auto sStr = s.toString();
    ASSERT_EQ(sStr, str);
    std::stringstream ss;
    ss << sStr;
    ASSERT_EQ(sStr, ss.str());
  };

  checkShapeStrEqual(Shape({3, 4, 7, 9}), "(3, 4, 7, 9)");
  checkShapeStrEqual(Shape({}), "()");
  checkShapeStrEqual(Shape({0}), "(0)");
  checkShapeStrEqual(Shape({7, 7, 7, 7, 7, 7, 7}), "(7, 7, 7, 7, 7, 7, 7)");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
