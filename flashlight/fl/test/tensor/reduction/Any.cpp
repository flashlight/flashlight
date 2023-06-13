/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

using namespace ::testing;
using namespace fl;

TEST(TensorOpAny, Base) {
  using fl::dtype;
  auto t = Tensor::fromVector<unsigned>({3, 3}, {1, 0, 0, 0, 0, 0, 0, 0, 1});
  auto anyAll = fl::any(t);
  ASSERT_EQ(anyAll.shape(), Shape({}));
  ASSERT_EQ(anyAll.elements(), 1);
  ASSERT_TRUE(anyAll.scalar<char>());
  ASSERT_TRUE(allClose(
      fl::any(t, {0}),
      Tensor::fromVector<unsigned>({1, 0, 1}).astype(dtype::b8)));
  ASSERT_TRUE(allClose(fl::any(t, {0, 1}), fl::fromScalar(true, dtype::b8)));
  ASSERT_FALSE(fl::any(Tensor::fromVector<unsigned>({0, 0, 0})).scalar<char>());

  auto keptDims = fl::any(
      fl::any(t, {1}, /* keepDims = */ true), {0}, /* keepDims = */ true);
  ASSERT_EQ(keptDims.shape(), Shape({1, 1}));
  ASSERT_EQ(keptDims.scalar<char>(), fl::any(t, {0, 1}).scalar<char>());
  auto q = fl::any(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(q.shape(), Shape({}));
  ASSERT_EQ(q.elements(), 1);
  ASSERT_EQ(q.scalar<char>(), true);

  const float v = 3.14;
  auto r = fl::any(fl::fromScalar(v));
  ASSERT_EQ(r.shape(), Shape());
  ASSERT_TRUE(r.scalar<char>());
  ASSERT_EQ(fl::any(fl::fromScalar(v), {0}).shape(), Shape());
}

TEST(TensorOpAny, Tensor1D) {
  auto t = Tensor::fromVector<unsigned>({0, 0, 0});
  auto res = fl::any(t);
  ASSERT_EQ(res.shape(), Shape());
  ASSERT_EQ(res.elements(), 1);
  ASSERT_TRUE(allClose(res, fl::fromScalar(false, dtype::b8)));

  t = Tensor::fromVector<unsigned>({1, 0, 1});
  res = fl::any(t);
  ASSERT_EQ(res.shape(), Shape());
  ASSERT_EQ(res.elements(), 1);
  ASSERT_TRUE(allClose(res, fl::fromScalar(true, dtype::b8)));

  t = Tensor::fromVector<unsigned>({1, 1, 1});
  res = fl::any(t);
  ASSERT_EQ(res.shape(), Shape());
  ASSERT_EQ(res.elements(), 1);
  ASSERT_TRUE(allClose(res, fl::fromScalar(true, dtype::b8)));
}

TEST(TensorOpAny, Tensor2D) {
  auto t2d_a = Tensor::fromVector<unsigned>({2, 2}, {1, 1, 0, 0});
  auto res = fl::any(t2d_a);
  ASSERT_EQ(res.shape(), Shape());
  ASSERT_EQ(res.elements(), 1);
  ASSERT_TRUE(allClose(res, fl::fromScalar(true, dtype::b8)));

  auto t2d_b = Tensor::fromVector<unsigned>({3, 2}, {1, 1, 0, 0, 1, 0});
  res = fl::any(t2d_b, {0, 1});
  ASSERT_EQ(res.shape(), Shape());
  ASSERT_EQ(res.elements(), 1);
  ASSERT_TRUE(allClose(res, fl::fromScalar(true, dtype::b8)));

  res = fl::any(t2d_a, {1});
  ASSERT_EQ(res.shape(), Shape({2}));
  ASSERT_EQ(res.elements(), 2);
  ASSERT_TRUE(
      allClose(res, Tensor::fromVector<unsigned>({1, 1}).astype(dtype::b8)));

  res = fl::any(t2d_a, {0});
  ASSERT_EQ(res.shape(), Shape({2}));
  ASSERT_EQ(res.elements(), 2);
  ASSERT_TRUE(
      allClose(res, Tensor::fromVector<unsigned>({1, 0}).astype(dtype::b8)));

  res = fl::any(t2d_b, {1}, true);
  ASSERT_EQ(res.shape(), Shape({3, 1}));
  ASSERT_EQ(res.elements(), 3);
  ASSERT_TRUE(allClose(
      res, Tensor::fromVector<unsigned>({3, 1}, {1, 1, 0}).astype(dtype::b8)));

  res = fl::any(t2d_b, {0});
  ASSERT_EQ(res.shape(), Shape({2}));
  ASSERT_EQ(res.elements(), 2);
  ASSERT_TRUE(
      allClose(res, Tensor::fromVector<unsigned>({1, 1}).astype(dtype::b8)));
}

TEST(TensorOpAny, IgnoresNAN) {
  auto t = Tensor::fromVector<float>({1, NAN, 1});
  auto res = fl::any(t);
  ASSERT_EQ(res.shape(), Shape());
  ASSERT_EQ(res.elements(), 1);
  ASSERT_TRUE(allClose(res, fl::fromScalar(true, dtype::b8)));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}