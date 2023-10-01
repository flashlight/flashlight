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

TEST(TensorOpNorm, Base) {
  auto r = fl::full({7, 8, 9}, 1);
  auto normAll = fl::norm(r);
  ASSERT_FLOAT_EQ(normAll.scalar<float>(), std::sqrt(7 * 8 * 9));
  ASSERT_EQ(normAll.shape(), Shape({}));
  ASSERT_EQ(normAll.elements(), 1);
  ASSERT_FLOAT_EQ(
      fl::norm(fl::full({5, 5}, 1.)).scalar<float>(), std::sqrt(5 * 5));
  ASSERT_EQ(
      fl::norm(r, {0, 1}, /* p = */ 2, /* keepDims = */ true).shape(),
      Shape({1, 1, 9}));

  ASSERT_FLOAT_EQ(fl::norm(r, {0}).scalar<float>(), std::sqrt(7));

  const float v = 3.14;
  auto q = fl::norm(fl::fromScalar(v));
  ASSERT_EQ(q.shape(), Shape());
  ASSERT_NEAR(q.scalar<float>(), 3.14, 1e-4);
  ASSERT_EQ(fl::norm(fl::fromScalar(v), {0}).shape(), Shape());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}