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

TEST(TensorOpMin, Base) {
  Tensor in = Tensor::fromVector<float>({2, 3}, {4, 8, 6, 3, 5, 9});
  Tensor values, indices;
  fl::min(values, indices, in, 0);
  ASSERT_EQ(indices.shape(), Shape({in.dim(1)}));
  ASSERT_TRUE(allClose(indices, Tensor::fromVector<unsigned>({3}, {0, 1, 0})));
  for (unsigned i = 0; i < values.elements(); ++i) {
    ASSERT_TRUE(allClose(values.flat(i), in(fl::span, i)(indices(i))));
  }

  fl::min(values, indices, in, 1);
  ASSERT_EQ(indices.shape(), Shape({in.dim(0)}));
  ASSERT_TRUE(allClose(indices, Tensor::fromVector<unsigned>({2}, {0, 1})));
  for (unsigned i = 0; i < values.elements(); ++i) {
    ASSERT_TRUE(allClose(values.flat(i), in(i)(indices(i))));
  }

  fl::min(values, indices, in, 0, /* keepDims = */ true);
  ASSERT_EQ(values.shape(), Shape({1, in.dim(1)}));

  fl::min(values, indices, in, 1, /* keepDims = */ true);
  ASSERT_EQ(values.shape(), Shape({in.dim(0), 1}));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}