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

TEST(TensorOpArgmin, Base) {
  Tensor in = Tensor::fromVector<float>({2, 3}, {4, 8, 6, 3, 5, 9});
  auto a0 = fl::argmin(in, 0);
  auto a1 = fl::argmin(in, 1);

  ASSERT_EQ(a0.shape(), Shape({in.dim(1)}));
  ASSERT_EQ(a1.shape(), Shape({in.dim(0)}));
  ASSERT_TRUE(allClose(a0, Tensor::fromVector<unsigned>({3}, {0, 1, 0})));
  ASSERT_TRUE(allClose(a1, Tensor::fromVector<unsigned>({2}, {0, 1})));
  ASSERT_EQ(
      fl::argmin(in, 0, /* keepDims = */ true).shape(), Shape({1, in.dim(1)}));
  ASSERT_EQ(
      fl::argmin(in, 1, /* keepDims = */ true).shape(), Shape({in.dim(0), 1}));
}

TEST(TensorOpArgmin, Scalar) {
  auto t = fl::fromScalar<unsigned>(10);
  auto res = fl::argmin(t, 0);
  ASSERT_TRUE(allClose(res, fl::fromScalar<unsigned>(0)));
}

TEST(TensorOpArgmin, Tensor1D) {
  auto t = Tensor::fromVector<unsigned>({1, 0, 3, 2});
  auto res = fl::argmin(t, 0);
  ASSERT_EQ(res.shape(), Shape());
  ASSERT_EQ(res.elements(), 1);
  ASSERT_TRUE(allClose(res, fl::fromScalar<unsigned>(1)));

  std::vector<unsigned> vec;
  vec.reserve(100);
  for (unsigned i = 0; i < 100; ++i) {
    vec.push_back(100 - i);
  }
  t = Tensor::fromVector<unsigned>(vec);
  res = fl::argmin(t, 0);
  ASSERT_TRUE(allClose(res, fl::fromScalar<unsigned>(99)));
}

TEST(TensorOpArgmin, Tensor2D) {
  auto t = Tensor::fromVector<float>({3, 2}, {3, -1, 0, 100, -7, 2});
  auto res = fl::argmin(t, 1);
  ASSERT_EQ(res.shape(), Shape({3}));
  ASSERT_TRUE(allClose(res, Tensor::fromVector<unsigned>({0, 1, 0})));

  t = Tensor::fromVector<float>({3, 2}, {3, 2, 5, 100, -7, -8});
  res = fl::argmin(t, 0);
  ASSERT_TRUE(allClose(res, Tensor::fromVector<unsigned>({1, 2})));
}

TEST(TensorOpArgmin, Tensor3D) {
  auto t = Tensor::fromVector<float>({5, 1, 1}, {5, 0, NAN, -1, 3});
  auto res = fl::argmin(t, 0);
  ASSERT_TRUE(allClose(res, Tensor::fromVector<unsigned>({1, 1}, {3})));
}

TEST(TensorOpArgmin, Tensor4D) {
  std::vector<unsigned> vec4d;
  vec4d.reserve(100);
  for (unsigned i = 0; i < 100; ++i) {
    vec4d.push_back(100 - i);
  }
  auto t = Tensor::fromVector<unsigned>({100, 1, 1, 1}, vec4d);
  auto res = fl::argmin(t, 0);
  ASSERT_TRUE(allClose(res, Tensor::fromVector<unsigned>({1, 1, 1}, {99})));
}

TEST(TensorOpArgmin, IgnoresNAN) {
  auto t = Tensor::fromVector<float>({5, 0, NAN, -1, 3});
  auto res = fl::argmin(t, 0);
  ASSERT_TRUE(allClose(res, fl::fromScalar<unsigned>(3)));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
