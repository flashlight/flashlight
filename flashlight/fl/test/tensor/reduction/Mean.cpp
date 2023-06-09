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

TEST(TensorOpMean, Base) {
  auto r = fl::rand({8, 7, 6});
  ASSERT_NEAR(fl::mean(r).scalar<float>(), 0.5, 0.05);
  ASSERT_EQ(
      fl::mean(r, {0, 1}, /* keepDims = */ true).shape(), Shape({1, 1, 6}));

  auto s = fl::full({5, 6, 7}, 1);
  ASSERT_TRUE(allClose(fl::mean(s, {0}), fl::full({6, 7}, 1.)));

  auto a = fl::mean(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(a.shape(), Shape({}));
  ASSERT_EQ(a.elements(), 1);
  ASSERT_EQ(a.scalar<float>(), 1.);

  // TODO: fixture this
  const float v = 3.14;
  auto q = fl::mean(fl::fromScalar(v));
  ASSERT_EQ(q.shape(), Shape());
  ASSERT_EQ(q.scalar<float>(), v);
  ASSERT_EQ(fl::mean(fl::fromScalar(v), {0}).shape(), Shape());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}