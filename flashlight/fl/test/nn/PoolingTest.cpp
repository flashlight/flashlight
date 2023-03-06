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

TEST(ModuleTest, PoolingFwd) {
  // test batching
  auto pool = Pool2D(9, 7, 1, 1, PaddingMode::SAME, PaddingMode::SAME);
  int batchsize = 10;
  auto input = fl::rand({120, 100, 30, batchsize});
  auto batchOutVar = pool(Variable(input, false));
  for (int i = 0; i < batchsize; ++i) {
    ASSERT_EQ(input.shape(), batchOutVar.shape());
    auto expected_outVar = pool(
        Variable(input(fl::span, fl::span, fl::span, fl::range(i, i + 1)), false));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-7));
  }
}

TEST_F(ModuleTestF16, PoolingFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // test batching
  auto pool = Pool2D(9, 7, 1, 1, PaddingMode::SAME, PaddingMode::SAME);
  int batchsize = 10;
  auto input = fl::rand({120, 100, 30, batchsize}, fl::dtype::f16);
  auto batchOutVar = pool(Variable(input, false));
  for (int i = 0; i < batchsize; ++i) {
    ASSERT_EQ(input.shape(), batchOutVar.shape());
    auto expected_outVar = pool(
        Variable(input(fl::span, fl::span, fl::span, fl::range(i, i + 1)), false));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-7));
  }
}
