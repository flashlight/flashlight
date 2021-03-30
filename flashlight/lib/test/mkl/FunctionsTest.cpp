/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/lib/mkl/Functions.h"

using namespace ::fl::lib;
using ::testing::ElementsAre;
using ::testing::Pointwise;

MATCHER_P(FloatNearPointwise, tol, "Out of range") {
  return (
      std::get<0>(arg) > std::get<1>(arg) - tol &&
      std::get<0>(arg) < std::get<1>(arg) + tol);
}

TEST(CorrelateTest, Identity) {
  std::vector<float> kernel = {1};
  std::vector<float> input = {1, 2, 3, 4, 5};
  std::vector<float> output = fl::lib::mkl::Correlate(kernel, input);
  EXPECT_EQ(output.size(), input.size() + kernel.size() - 1);
  EXPECT_THAT(output, Pointwise(FloatNearPointwise(0.01), input));
}

TEST(CorrelateTest, BasicReverb) {
  std::vector<float> kernel = {1, 2, 3};
  std::vector<float> input = {1, 2, 3, 4, 5};
  std::vector<float> output = fl::lib::mkl::Correlate(kernel, input);
  EXPECT_EQ(output.size(), input.size() + kernel.size() - 1);
  EXPECT_THAT(output, ElementsAre(1, 4, 10, 16, 22, 22, 15));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
