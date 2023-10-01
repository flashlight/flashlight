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

TEST(TensorOpMedian, Base) {
  auto a = Tensor::fromVector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_EQ(fl::median(a).scalar<float>(), 4.5);
  ASSERT_TRUE(allClose(fl::median(a, {0}), fl::fromScalar(4.5)));
  ASSERT_EQ(fl::median(fl::rand({5, 6, 7, 8}), {1, 2}).shape(), Shape({5, 8}));
  ASSERT_EQ(
      fl::median(fl::rand({5, 6, 7, 8}), {1, 2}, /* keepDims = */ true).shape(),
      Shape({5, 1, 1, 8}));

  auto b = fl::median(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(b.shape(), Shape({}));
  ASSERT_EQ(b.elements(), 1);
  ASSERT_EQ(b.scalar<float>(), 1.);

  const float v = 3.14;
  auto q = fl::median(fl::fromScalar(v));
  ASSERT_EQ(q.shape(), Shape());
  ASSERT_EQ(q.scalar<float>(), v);
  ASSERT_EQ(fl::median(fl::fromScalar(v), {0}).shape(), Shape());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}