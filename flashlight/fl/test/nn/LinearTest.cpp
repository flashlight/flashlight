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

TEST(ModuleTest, LinearFwd) {
  int n_in = 2, n_out = 3, x = 4, batchsize = 2;
  auto wtVar =
      param(Tensor::fromVector<float>({n_out, n_in}, {8, 2, 2, 10, 5, 3}));

  auto inVar = input(Tensor::fromVector<float>(
      {n_in, x, batchsize}, {6, 2, 1, 4, 8, 2, 7, 1, 10, 7, 3, 7, 5, 9, 2, 4}));

  auto expected_outVar = Variable(
      Tensor::fromVector<float>(
          {n_out, x, batchsize},
          {68,  22, 18, 48, 22, 14, 84,  26, 22, 66, 19, 17,
           150, 55, 41, 94, 41, 27, 130, 55, 37, 56, 24, 16}),
      true);

  auto linNoBias = Linear(wtVar);
  ASSERT_TRUE(allClose(linNoBias.forward(inVar), expected_outVar, 1E-7));

  auto bsVar = input(Tensor::fromVector<float>({n_out}, {1, 2, 3}));
  expected_outVar = Variable(
      Tensor::fromVector<float>(
          {n_out, x, batchsize},
          {69,  24, 21, 49, 24, 17, 85,  28, 25, 67, 21, 20,
           151, 57, 44, 95, 43, 30, 131, 57, 40, 57, 26, 19}),
      true);

  auto linBias = Linear(wtVar, bsVar);
  ASSERT_TRUE(allClose(linBias.forward(inVar), expected_outVar, 1E-7));
}

TEST_F(ModuleTestF16, LinearFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  int n_in = 2, n_out = 3, x = 4, batchsize = 2;
  auto wtVar =
      param(Tensor::fromVector<float>({n_out, n_in}, {8, 2, 2, 10, 5, 3}));

  auto inVar = input(Tensor::fromVector<float>(
                         {n_in, x, batchsize},
                         {6, 2, 1, 4, 8, 2, 7, 1, 10, 7, 3, 7, 5, 9, 2, 4})
                         .astype(fl::dtype::f16));

  auto expected_outVar = Variable(
      Tensor::fromVector<float>(
          {n_out, x, batchsize},
          {68,  22, 18, 48, 22, 14, 84,  26, 22, 66, 19, 17,
           150, 55, 41, 94, 41, 27, 130, 55, 37, 56, 24, 16})
          .astype(fl::dtype::f16),
      true);

  auto linNoBias = Linear(wtVar);
  auto result = linNoBias.forward(inVar);
  ASSERT_EQ(result.type(), inVar.type());
  ASSERT_TRUE(allClose(result, expected_outVar, 1E-2));

  auto bsVar = input(Tensor::fromVector<float>({n_out}, {1, 2, 3}));
  ;
  expected_outVar = Variable(
      Tensor::fromVector<float>(
          {n_out, x, batchsize},
          {69,  24, 21, 49, 24, 17, 85,  28, 25, 67, 21, 20,
           151, 57, 44, 95, 43, 30, 131, 57, 40, 57, 26, 19})
          .astype(inVar.type()),
      true);

  auto linBias = Linear(wtVar, bsVar);
  auto resultBias = linBias.forward(inVar);
  ASSERT_EQ(resultBias.type(), fl::dtype::f16);
  ASSERT_TRUE(allClose(resultBias, expected_outVar, 1E-3));

  // OptimLevel::O3 is active with this fixture
  ASSERT_EQ(linBias.forward(inVar.astype(fl::dtype::f32)).type(), fl::dtype::f16);
}
