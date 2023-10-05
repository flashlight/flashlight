/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "flashlight/pkg/vision/nn/PositionalEmbeddingSine.h"

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"

using namespace fl;
using namespace fl::pkg::vision;

TEST(PositionalEmbeddingSine, PytorchComparision) {
  int hiddenDim = 8;
  int H = 6;
  int W = 6;
  int B = 1;
  Shape dims = {W, H, 1, B};
  auto inputArray = fl::full(dims, 0);
  inputArray(fl::range(0, 4), fl::range(0, 4)) = fl::full({4, 4, 1, B}, 1);
  auto input = Variable(inputArray, false);

  PositionalEmbeddingSine pos(hiddenDim / 2, 10000.0f, false, 0.0f);

  auto result = pos.forward({input})[0];
  ASSERT_EQ(result.shape(), Shape({6, 6, 8, 1}));
  ASSERT_LE(result(0, 5, 3, 0).tensor().scalar<float>() - 0.9992f, 1e-5);
  ASSERT_LE(result(0, 0, 0, 0).tensor().scalar<float>() - 0.841471f, 1e-5);
}
