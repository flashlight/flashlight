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

TEST(ModuleTest, GLUFwd) {
  auto inVar = Variable(
      Tensor::fromVector<float>({3, 2}, {0.8, 0.2, 0.2, 0.1, 0.5, 0.3}), true);

  auto expected_outVar = Variable(
      Tensor::fromVector<float>({3, 1}, {0.419983, 0.124492, 0.114888}), true);

  GatedLinearUnit glu(1);
  ASSERT_TRUE(allClose(glu.forward(inVar), expected_outVar, 1E-4));

  // test batching
  int batchsize = 5;
  inVar = Variable(fl::rand({10, 7, batchsize}), true);
  glu = GatedLinearUnit(0);

  auto batchOutVar = glu(inVar);

  for (int i = 0; i < batchsize; ++i) {
    expected_outVar = glu.forward(inVar(fl::span, fl::span, fl::range(i, i + 1)));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-7));
  }
}

TEST_F(ModuleTestF16, GLUFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto inVar = Variable(
      Tensor::fromVector<float>({3, 2}, {0.8, 0.2, 0.2, 0.1, 0.5, 0.3})
          .astype(fl::dtype::f16),
      true);

  auto expected_outVar = Variable(
      Tensor::fromVector<float>({3, 1}, {0.419983, 0.124492, 0.114888})
          .astype(fl::dtype::f16),
      true);

  GatedLinearUnit glu(1);
  auto out = glu.forward(inVar);
  ASSERT_EQ(out.type(), inVar.type());
  ASSERT_TRUE(allClose(out, expected_outVar, 1E-2));

  // test batching
  int batchsize = 5;
  inVar = Variable(fl::rand({10, 7, batchsize}).astype(fl::dtype::f16), true);
  glu = GatedLinearUnit(0);

  auto batchOutVar = glu(inVar);

  for (int i = 0; i < batchsize; ++i) {
    expected_outVar = glu.forward(inVar(fl::span, fl::span, fl::range(i, i + 1)));
    ASSERT_EQ(batchOutVar.type(), expected_outVar.type());
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-3));
  }
}

TEST(ModuleTest, LogSoftmaxFwd) {
  auto inVar = Variable(
      Tensor::fromVector<float>({3, 2}, {0.8, 0.2, 0.2, 0.1, 0.5, 0.3}), true);

  auto expected_outVar0 = Variable(
      Tensor::fromVector<float>(
          {3, 2}, {-0.740805, -1.34081, -1.34081, -1.3119, -0.911902, -1.1119}),
      true);
  LogSoftmax lsm0(0);
  ASSERT_TRUE(allClose(lsm0.forward(inVar), expected_outVar0, 1E-4));

  auto expected_outVar1 = Variable(
      Tensor::fromVector<float>(
          {3, 2},
          {-0.403186, -0.854355, -0.744397, -1.10319, -0.554355, -0.644397}),
      true);
  LogSoftmax lsm1(1);
  ASSERT_TRUE(allClose(lsm1.forward(inVar), expected_outVar1, 1E-4));

  // test batching
  int batchsize = 5;
  inVar = Variable(fl::rand({10, 7, batchsize}), true);
  LogSoftmax lsm(0);

  auto batchOutVar = lsm(inVar);

  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar =
        lsm.forward(inVar(fl::span, fl::span, fl::range(i, i + 1)));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-7));
  }
}

TEST_F(ModuleTestF16, LogSoftmaxFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto inVar = Variable(
      Tensor::fromVector<float>({3, 2}, {0.8, 0.2, 0.2, 0.1, 0.5, 0.3})
          .astype(fl::dtype::f16),
      true);

  auto expected_outVar0 = Variable(
      Tensor::fromVector<float>(
          {3, 2}, {-0.740805, -1.34081, -1.34081, -1.3119, -0.911902, -1.1119}),
      true);
  LogSoftmax lsm0(0);
  auto result0 = lsm0.forward(inVar);
  ASSERT_TRUE(allClose(result0, expected_outVar0, 1E-3));

  auto expected_outVar1 = Variable(
      Tensor::fromVector<float>(
          {3, 2},
          {-0.403186, -0.854355, -0.744397, -1.10319, -0.554355, -0.644397}),
      true);
  LogSoftmax lsm1(1);
  ASSERT_TRUE(allClose(lsm1.forward(inVar), expected_outVar1, 1E-3));

  // test batching
  int batchsize = 5;
  inVar = Variable(fl::rand({10, 7, batchsize}), true);
  LogSoftmax lsm(0);

  auto batchOutVar = lsm(inVar);

  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar =
        lsm.forward(inVar(fl::span, fl::span, fl::range(i, i + 1)));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-7));
  }
}
