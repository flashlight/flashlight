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

TEST(TensorOpVar, Base) {
  auto r = fl::rand({7, 8, 9});
  auto varAll = fl::var(r);
  ASSERT_NEAR(varAll.scalar<float>(), 0.08333, 0.01);
  ASSERT_EQ(varAll.shape(), Shape({}));
  ASSERT_EQ(varAll.elements(), 1);

  ASSERT_EQ(
      fl::var(r, {0, 1}, /* bias = */ false, /* keepDims = */ true).shape(),
      Shape({1, 1, 9}));

  auto s = fl::full({5, 6, 7}, 1);
  ASSERT_TRUE(allClose(fl::var(s, {0}), fl::full({6, 7}, 0.)));
  auto a = fl::rand({5, 5});
  ASSERT_TRUE(allClose(fl::var(a), fl::var(a, {0, 1})));

  const float v = 3.14;
  auto q = fl::var(fl::fromScalar(v));
  ASSERT_EQ(q.shape(), Shape());
  ASSERT_EQ(q.scalar<float>(), 0);
  ASSERT_EQ(fl::var(fl::fromScalar(v), {0}).shape(), Shape());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}