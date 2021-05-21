/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "flashlight/pkg/vision/nn/PositionalEmbeddingSine.h"

#include <gtest/gtest.h>

using namespace fl;
using namespace fl::app::objdet;

TEST(PositionalEmbeddingSine, PytorchComparision) {
  int hiddenDim = 8;
  int H = 6;
  int W = 6;
  int B = 1;
  af::dim4 dims = {W, H, 1, B};
  auto inputArray = af::constant(0, dims);
  inputArray(af::seq(0, 3), af::seq(0, 3)) = af::constant(1, {4, 4, B});
  auto input = Variable(inputArray, false);

  PositionalEmbeddingSine pos(hiddenDim / 2, 10000.0f, false, 0.0f);

  auto result = pos.forward({input})[0];
  EXPECT_LE(result(0, 5, 3, 0).array().scalar<float>() - 0.9992f, 1e-5);
  EXPECT_LE(result(0, 0, 0, 0).array().scalar<float>() - 0.841471f, 1e-5);
}
