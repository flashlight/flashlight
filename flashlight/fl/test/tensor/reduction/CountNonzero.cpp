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

TEST(TensorOpCountNonzero, Base) {
  std::vector<int> idxs = {0, 3, 4, 7, 24, 78};
  auto a = fl::full({10, 10}, 1, fl::dtype::u32);
  for (const auto idx : idxs) {
    a(idx / 10, idx % 10) = 0;
  }

  ASSERT_TRUE(allClose(
      fl::fromScalar(a.elements() - idxs.size()), fl::countNonzero(a)));

  std::vector<unsigned> sizes(a.shape().dim(0));
  for (unsigned i = 0; i < a.shape().dim(0); ++i) {
    sizes[i] =
        a.shape().dim(0) - fl::sum(a(fl::span, i) == 0, {0}).scalar<unsigned>();
  }
  ASSERT_TRUE(allClose(Tensor::fromVector(sizes), Tensor::fromVector(sizes)));

  auto b = fl::full({2, 2, 2}, 1, fl::dtype::u32);
  b(0, 1, 1) = 0;
  b(1, 0, 1) = 0;
  b(1, 1, 1) = 0;
  ASSERT_TRUE(allClose(
      fl::Tensor::fromVector<unsigned>({2, 2}, {2, 2, 1, 0}),
      fl::countNonzero(b, {0})));
  ASSERT_TRUE(allClose(
      fl::Tensor::fromVector<unsigned>({2}, {4, 1}),
      fl::countNonzero(b, {0, 1})));
  ASSERT_TRUE(allClose(
      fl::fromScalar(b.elements() - 3), fl::countNonzero(b, {0, 1, 2})));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
