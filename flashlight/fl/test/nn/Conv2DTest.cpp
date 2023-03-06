/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

using namespace fl;

namespace {

class ModuleTestF16 : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensures all operations will be in f16
    OptimMode::get().setOptimLevel(OptimLevel::O3);
  }

  void TearDown() override {
    OptimMode::get().setOptimLevel(OptimLevel::DEFAULT);
  }
};

} // namespace

TEST(ModuleTest, ConvPadding) {
  auto conv1 = Conv2D(30, 100, 3, 5, 2, 1, PaddingMode::SAME, 0, true, 1);
  auto conv2 = Conv2D(
      30, 100, 3, 5, 2, 1, PaddingMode::SAME, PaddingMode::SAME, true, 1);
  auto conv3 =
      Conv2D(30, 100, 10, 10, 1, 1, PaddingMode::SAME, PaddingMode::SAME, 4, 4);
  auto input = Variable(fl::rand({32, 32, 30, 2}), false);

  auto conv1Op = conv1(input);
  ASSERT_EQ(conv1Op.shape(), Shape({16, 28, 100, 2}));

  auto conv2Op = conv2(input);
  ASSERT_EQ(conv2Op.shape(), Shape({16, 32, 100, 2}));

  // test dilation
  auto conv3Op = conv3(input);
  ASSERT_EQ(conv3Op.shape(), Shape({32, 32, 100, 2}));
}

TEST(ModuleTest, ConvolutionFwd) {
  // test batching
  auto conv = Conv2D(30, 50, 9, 7, 2, 3, 3, 2, 1, 1, true, 1);
  int batchsize = 10;
  auto input = fl::rand({120, 100, 30, batchsize});
  auto batchOutVar = conv(Variable(input, false));

  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar = conv(
        Variable(input(fl::span, fl::span, fl::span, fl::range(i, i + 1)), false));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-5));
  }
}

TEST_F(ModuleTestF16, ConvolutionFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // test batching
  auto conv = Conv2D(30, 50, 9, 7, 2, 3, 3, 2, 1, 1, true, 1);
  int batchsize = 1;
  auto input = fl::rand({120, 100, 30, batchsize}, fl::dtype::f16);
  auto batchOutVar = conv(Variable(input, false));
  ASSERT_EQ(batchOutVar.type(), input.type());

  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar = conv(
        Variable(input(fl::span, fl::span, fl::span, fl::range(i, i + 1)), false));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-7));
  }

  auto inputF32 = fl::rand({120, 100, 30, batchsize}, fl::dtype::f32);
  ASSERT_EQ(
      conv(Variable(input, false)).type(),
      fl::dtype::f16); // OptimLevel::O3 is active with this fixture
}

TEST(ModuleTest, ConvolutionWithGroupFwd) {
  // test batching
  auto conv = Conv2D(30, 50, 9, 7, 2, 3, 3, 2, true, 2);
  int batchsize = 10;
  auto input = fl::rand({120, 100, 30, batchsize});
  auto batchOutVar = conv(Variable(input, false));
  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar = conv(
        Variable(input(fl::span, fl::span, fl::span, fl::range(i, i + 1)), false));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-5));
  }
}
