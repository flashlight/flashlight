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

TEST(TensorOpStd, Base) {
  auto r = fl::rand({7, 8, 9});
  ASSERT_NEAR(fl::std(r).scalar<float>(), 0.2886, 0.005);
  ASSERT_EQ(
      fl::std(r, {0, 1}, /* keepDims = */ true).shape(), Shape({1, 1, 9}));

  auto s = fl::full({5, 6, 7}, 1);
  ASSERT_TRUE(allClose(fl::std(s, {0}), fl::full({6, 7}, 0.)));
  ASSERT_TRUE(allClose(fl::std(s, {1}), fl::sqrt(fl::var(s, {1}))));

  const float v = 3.14;
  auto q = fl::std(fl::fromScalar(v));
  ASSERT_EQ(q.shape(), Shape());
  ASSERT_EQ(q.scalar<float>(), 0);
  ASSERT_EQ(fl::std(fl::fromScalar(v), {0}).shape(), Shape());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}