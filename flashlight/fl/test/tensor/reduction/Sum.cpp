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

TEST(TensorOpSum, Base) {
  auto t = fl::full({3, 4, 5, 6}, 1.0);
  ASSERT_TRUE(allClose(fl::sum(t, {0}), fl::full({4, 5, 6}, 3.0)));
  ASSERT_TRUE(
      allClose(fl::sum(t, {1, 2}), fl::full({3, 6}, 4 * 5, fl::dtype::f32)));
  auto res = fl::sum(
      fl::sum(t, {2}, /* keepDims = */ true), {1}, /* keepDims = */ true);
  ASSERT_EQ(res.shape(), Shape({t.dim(0), 1, 1, t.dim(3)}));
  ASSERT_TRUE(
      allClose(fl::reshape(res, {t.dim(0), t.dim(3)}), fl::sum(t, {2, 1})));

  unsigned dim = 5;
  auto q = fl::sum(fl::full({dim, dim, dim, dim}, 1));
  ASSERT_EQ(q.shape(), Shape({}));
  ASSERT_EQ(q.elements(), 1);
  ASSERT_EQ(q.scalar<int>(), dim * dim * dim * dim);

  ASSERT_TRUE(allClose(
      fl::sum(fl::sum(q, {0, 1, 2}), {0}),
      fl::fromScalar(dim * dim * dim * dim, fl::dtype::s32)));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}