/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBase.h"

using namespace ::testing;
using namespace fl;

TEST(TensorBaseTest, DefaultBackend) {
  Tensor t;
  ASSERT_EQ(t.backendType(), TensorBackendType::ArrayFire);
}

TEST(TensorBaseTest, BinaryOperators) {
  // TODO:fl::Tensor {testing} expand this test
  // Ensure that some binary operators work properly.
  auto a = fl::full({2, 2}, 1);
  auto b = fl::full({2, 2}, 2);
  auto c = fl::full({2, 2}, 3);

  ASSERT_TRUE(allClose((a == b), (b == c)));
  ASSERT_TRUE(allClose((a + b), c));
  ASSERT_TRUE(allClose((c - b), a));
  ASSERT_TRUE(allClose((c * b), fl::full({2, 2}, 6)));
}

TEST(TensorBaseTest, AssignmentOperators) {
  auto a = fl::full({3, 3}, 1.);
  a += 2;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 3.)));
  a -= 1;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 2.)));
  a *= 8;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 16.)));
  a /= 4;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 4.)));

  a = fl::full({4, 4}, 7.);
  ASSERT_TRUE(allClose(a, fl::full({4, 4}, 7.)));
  auto b = a;
  ASSERT_TRUE(allClose(b, fl::full({4, 4}, 7.)));
  a = 6.;
  ASSERT_TRUE(allClose(a, fl::full({4, 4}, 6.)));
}

TEST(TensorBaseTest, reshape) {
  auto a = fl::full({4, 4}, 3.);
  auto b = fl::reshape(a, Shape({8, 2}));
  ASSERT_EQ(b.shape(), Shape({8, 2}));
  ASSERT_TRUE(allClose(a, fl::reshape(b, {4, 4})));
}

TEST(TensorBaseTest, transpose) {
  // TODO: expand to check els
  ASSERT_TRUE(
      allClose(fl::transpose(fl::full({3, 4}, 3.)), fl::full({4, 3}, 3.)));
  ASSERT_TRUE(allClose(
      fl::transpose(fl::full({4, 5, 6, 7}, 3.), {2, 0, 1}),
      fl::full({6, 4, 5, 7}, 3.)));
}

TEST(TensorBaseTest, tile) {
  auto a = fl::full({4, 4}, 3.);
  auto tiled = fl::tile(a, {2, 2});
  ASSERT_EQ(tiled.shape(), Shape({8, 8}));
  ASSERT_TRUE(allClose(tiled, fl::full({8, 8}, 3.)));
}

TEST(TensorBaseTest, concatenate) {
  auto a = fl::full({3, 3}, 1.);
  auto b = fl::full({3, 3}, 2.);
  auto c = fl::full({3, 3}, 3.);
  ASSERT_TRUE(
      allClose(fl::concatenate(0, a, b, c), fl::concatenate({a, b, c})));
  auto out = fl::concatenate(0, a, b, c);
  ASSERT_EQ(out.shape(), Shape({9, 3}));
}

TEST(TensorBaseTest, flatten) {
  unsigned s = 6;
  auto a = fl::full({s, s, s}, 2.);
  auto flat = a.flatten();
  ASSERT_EQ(flat.shape(), Shape({s * s * s}));
  ASSERT_TRUE(allClose(flat, fl::full({s * s * s}, 2.)));
}

TEST(TensorBaseTest, minimum) {
  auto a = fl::full({3, 3}, 1);
  auto b = fl::full({3, 3}, 2);
  auto c = fl::minimum(a, b);
  ASSERT_EQ(a.type(), c.type());
  ASSERT_TRUE(allClose(a, c));
  ASSERT_TRUE(allClose(fl::minimum(1, b).astype(a.type()), a));
  ASSERT_TRUE(allClose(fl::minimum(b, 1).astype(a.type()), a));
}

TEST(TensorBaseTest, maximum) {
  auto a = fl::full({3, 3}, 1);
  auto b = fl::full({3, 3}, 2);
  auto c = fl::maximum(a, b);
  ASSERT_EQ(b.type(), c.type());
  ASSERT_TRUE(allClose(b, c));
  ASSERT_TRUE(allClose(fl::maximum(1, b).astype(a.type()), b));
  ASSERT_TRUE(allClose(fl::maximum(b, 1).astype(a.type()), b));
}

TEST(TensorBaseTest, negative) {
  auto a = fl::full({3, 3}, 1);
  auto b = fl::full({3, 3}, 2);
  auto c = a - b;
  ASSERT_TRUE(allClose(c, -a));
  ASSERT_TRUE(allClose(c, negative(a)));
}

TEST(TensorBaseTest, astype) {
  auto a = fl::rand({3, 3});
  ASSERT_EQ(a.type(), dtype::f32);
  ASSERT_EQ(a.astype(dtype::f64).type(), dtype::f64);
}

TEST(TensorBaseTest, logicalNot) {
  ASSERT_TRUE(allClose(
      !fl::full({3, 3}, true), fl::full({3, 3}, false).astype(dtype::b8)));
}

TEST(TensorBaseTest, clip) {
  float h = 3.;
  float l = 2.;
  Shape s = {3, 3};
  auto high = fl::full(s, h);
  auto low = fl::full(s, l);
  ASSERT_TRUE(allClose(fl::clip(fl::full({3, 3}, 4.), low, high), high));
  ASSERT_TRUE(allClose(fl::clip(fl::full({3, 3}, 4.), l, high), high));
  ASSERT_TRUE(allClose(fl::clip(fl::full({3, 3}, 4.), low, h), high));
  ASSERT_TRUE(allClose(fl::clip(fl::full({3, 3}, 4.), l, h), high));
}

TEST(TensorBaseTest, isnan) {
  Shape s = {3, 3};
  ASSERT_TRUE(allClose(
      fl::isnan(fl::full(s, 1.) / 3),
      fl::full(s, false).astype(fl::dtype::b8)));
}

TEST(TensorBaseTest, power) {
  auto a = fl::full({3, 3}, 2.);
  auto b = fl::full({3, 3}, 2.);
  ASSERT_TRUE(allClose(fl::power(a, b), a * b));
}
