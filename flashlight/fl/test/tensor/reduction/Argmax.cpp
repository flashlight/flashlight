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

TEST(TensorOpArgmax, Base) {
  Tensor in = Tensor::fromVector<float>({2, 3}, {4, 8, 6, 3, 5, 9});
  auto a0 = fl::argmax(in, 0);
  auto a1 = fl::argmax(in, 1);

  ASSERT_EQ(a0.shape(), Shape({in.dim(1)}));
  ASSERT_EQ(a1.shape(), Shape({in.dim(0)}));
  ASSERT_TRUE(allClose(a0, Tensor::fromVector<unsigned>({3}, {1, 0, 1})));
  ASSERT_TRUE(allClose(a1, Tensor::fromVector<unsigned>({2}, {1, 2})));
  ASSERT_EQ(
      fl::argmax(in, 0, /* keepDims = */ true).shape(), Shape({1, in.dim(1)}));
  ASSERT_EQ(
      fl::argmax(in, 1, /* keepDims = */ true).shape(), Shape({in.dim(0), 1}));
}

TEST(TensorOpArgmax, Scalar) {
  auto t = fl::fromScalar<unsigned>(10);
  auto res = fl::argmax(t, 0);
  ASSERT_TRUE(allClose(res, fl::fromScalar<unsigned>(0)));
}

TEST(TensorOpArgmax, Tensor1D) {
  auto t = Tensor::fromVector<unsigned>({1, 0, 3, 2});
  auto res = fl::argmax(t, 0);
  ASSERT_EQ(res.shape(), Shape());
  ASSERT_EQ(res.elements(), 1);
  ASSERT_TRUE(allClose(res, fl::fromScalar<unsigned>(2)));

  std::vector<unsigned> vec;
  vec.reserve(100);
  for (unsigned i = 0; i < 100; ++i) {
    vec.push_back(i);
  }
  t = Tensor::fromVector<unsigned>(vec);
  res = fl::argmax(t, 0);
  ASSERT_TRUE(allClose(res, fl::fromScalar<unsigned>(99)));
}

TEST(TensorOpArgmax, Tensor2D) {
  auto t = Tensor::fromVector<float>({3, 2}, {3, -1, 0, 100, -7, 2});
  auto res = fl::argmax(t, 1);
  ASSERT_EQ(res.shape(), Shape({3}));
  ASSERT_TRUE(allClose(res, Tensor::fromVector<unsigned>({1, 0, 1})));

  t = Tensor::fromVector<float>({3, 2}, {3, 2, 5, 100, -7, 2});
  res = fl::argmax(t, 0);
  ASSERT_TRUE(allClose(res, Tensor::fromVector<unsigned>({2, 0})));
}

TEST(TensorOpArgmax, Tensor3D) {
  std::vector<float> vec3d;
  vec3d.reserve(60);
  for (float i = 0; i < 60; ++i) {
    vec3d.push_back(i);
  }
  auto t = Tensor::fromVector<float>({60, 1, 1}, vec3d);
  auto res = fl::argmax(t, 0);
  ASSERT_TRUE(allClose(res, Tensor::fromVector<unsigned>({1, 1}, {59})));
}

TEST(TensorOpArgmax, IgnoresNAN) {
  auto t = Tensor::fromVector<float>({0, 3, 5, NAN, 3});
  auto res = fl::argmax(t, 0);
  ASSERT_TRUE(allClose(res, fl::fromScalar<unsigned>(2)));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
