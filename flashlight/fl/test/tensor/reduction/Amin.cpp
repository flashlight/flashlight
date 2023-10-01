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

TEST(TensorOpAmin, Base) {
  auto a = fl::rand({4, 5, 6});
  const float val = -300;
  a(2, 3, 4) = val;
  ASSERT_EQ(fl::amin(a).shape(), Shape({}));
  ASSERT_EQ(fl::amin(a).elements(), 1);
  ASSERT_EQ(fl::amin(a).scalar<float>(), val);
  auto b = fl::rand({4, 4});
  b(1, 1) = val;
  ASSERT_EQ(fl::amin(b, {0}).shape(), Shape({4}));
  ASSERT_EQ(fl::amin(b, {0}, /* keepDims = */ true).shape(), Shape({1, 4}));
  ASSERT_EQ(fl::amin(b, {0})(1).scalar<float>(), val);
  ASSERT_EQ(fl::amin(b, {1})(1).scalar<float>(), val);
  auto q = fl::amin(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(q.shape(), Shape({}));
  ASSERT_EQ(q.elements(), 1);
  ASSERT_EQ(q.scalar<int>(), 1);

  const float v = 3.14;
  auto s = fl::amin(fl::fromScalar(v));
  ASSERT_EQ(s.shape(), Shape());
  ASSERT_EQ(s.scalar<float>(), v);
  ASSERT_EQ(fl::amin(fl::fromScalar(v), {0}).shape(), Shape());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}