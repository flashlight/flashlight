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

TEST(TensorOpCumsum, Base) {
  int max = 30;
  auto a = fl::tile(fl::arange(1, max), {1, 2});

  auto ref = fl::arange(1, max);
  for (int i = 1; i < max - 1; ++i) {
    ref += fl::concatenate({fl::full({i}, 0), fl::arange(1, max - i)});
  }

  ASSERT_TRUE(allClose(fl::cumsum(a, 0), fl::tile(ref, {1, 2})));
  ASSERT_TRUE(allClose(
      fl::cumsum(a, 1),
      fl::concatenate(
          {fl::arange(1, max), 2 * fl::arange(1, max)}, /* axis = */ 1)));
}

TEST(TensorOpCumsum, Tensor1D) {
  auto t = Tensor::fromVector<unsigned>({1, 2, 3, 4});
  auto res = fl::cumsum(t, 0);
  ASSERT_EQ(res.shape(), Shape({4}));
  ASSERT_TRUE(allClose(res, Tensor::fromVector<unsigned>({1, 3, 6, 10})));
}

TEST(TensorOpCumsum, Tensor2D) {
  auto t = Tensor::fromVector<unsigned>({2, 2}, {1, 2, 3, 4});
  auto res = fl::cumsum(t, 0);
  ASSERT_EQ(res.shape(), Shape({2, 2}));
  ASSERT_TRUE(
      allClose(res, Tensor::fromVector<unsigned>({2, 2}, {1, 3, 3, 7})));

  res = fl::cumsum(t, 1);
  ASSERT_EQ(res.shape(), Shape({2, 2}));
  ASSERT_TRUE(
      allClose(res, Tensor::fromVector<unsigned>({2, 2}, {1, 2, 4, 6})));
}

TEST(TensorOpCumsum, Tensor3D) {
  auto t = Tensor::fromVector<unsigned>({2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  auto res = fl::cumsum(t, 0);
  ASSERT_EQ(res.shape(), Shape({2, 2, 2}));
  ASSERT_TRUE(allClose(
      res, Tensor::fromVector<unsigned>({2, 2, 2}, {0, 1, 2, 5, 4, 9, 6, 13})));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}