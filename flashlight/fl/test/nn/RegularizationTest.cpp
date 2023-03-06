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

TEST(ModuleTest, DropoutFwd) {
  auto module = Dropout(0.5);
  // Train Mode
  module.train();
  auto in = Variable(fl::rand({1000, 1000}), true);
  auto out = module(in);

  ASSERT_NEAR(
      out.elements() - fl::countNonzero(out.tensor()).scalar<unsigned>(),
      in.elements() / 2,
      in.elements() / 16); // Check enough zeroes

  ASSERT_GT(
      fl::amax(out.tensor()).scalar<float>(), 1.5); // Check input is scaled

  // Eval Mode
  module.eval();
  out = module(in);
  ASSERT_TRUE(allClose(out, in, 1E-5));
}

TEST_F(ModuleTestF16, DropoutFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto module = Dropout(0.5);
  // Train Mode
  module.train();
  auto in = Variable(fl::rand({1000, 1000}, fl::dtype::f16), true);
  auto out = module(in);
  ASSERT_EQ(out.type(), fl::dtype::f16);

  ASSERT_NEAR(
      out.elements() - fl::countNonzero(out.tensor()).scalar<unsigned>(),
      in.elements() / 2,
      in.elements() / 16); // Check enough zeroes

  ASSERT_GT(
      fl::amax(out.tensor()).asScalar<float>(), 1.5); // Check input is scaled

  // Eval Mode
  module.eval();
  out = module(in);
  ASSERT_TRUE(allClose(out, in, 1E-5));
}

TEST(ModuleTest, LayerNormFwd) {
  double eps = 1E-5;
  std::vector<int> feat_axes = {3};
  int F = 10;
  auto input = Variable(fl::rand({4, 4, 3, F}), true);

  auto sample_mean = mean(input, {3});
  auto sample_var = var(input, {3}, true);
  auto true_out = (input - tileAs(sample_mean, input)) /
      tileAs(fl::sqrt(sample_var + eps), input);

  // no affine transform
  auto module1 = LayerNorm(feat_axes, eps, false);

  module1.train();
  auto out = module1.forward(input);

  ASSERT_TRUE(allClose(out, true_out, eps));
  ASSERT_EQ(out.type(), input.type());

  module1.eval();
  out = module1.forward(input);

  ASSERT_TRUE(allClose(out.tensor(), true_out.tensor(), eps));
  ASSERT_EQ(out.type(), input.type());

  // with affine transform
  auto module2 = LayerNorm(feat_axes, eps, true);

  module2.train();
  auto out_train = module2.forward(input);
  module2.eval();
  auto out_eval = module2.forward(input);

  ASSERT_TRUE(allClose(out_train.tensor(), out_eval.tensor(), eps));
  ASSERT_EQ(out_train.shape(), input.shape());

  // with affine transform
  auto module3 = LayerNorm(feat_axes, eps, true, F);
  module3.setParams(Variable(fl::full({F}, 1.0), false), 0);
  module3.setParams(Variable(fl::full({F}, 0.0), false), 1);
  auto out3 = module3.forward(input);
  ASSERT_TRUE(allClose(out_train.tensor(), out3.tensor(), eps));

  // With other shapes
  auto input3Dim = Variable(fl::rand({4, 4, 3}), true);
  auto module4 = LayerNorm(std::vector<int>{0}, eps, false);
  out = module4.forward(input3Dim);
  ASSERT_EQ(out.shape(), input3Dim.shape());
}

TEST_F(ModuleTestF16, LayerNormFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  double eps = 5E-2;
  std::vector<int> feat_axes = {3};
  auto input = Variable(fl::rand({4, 4, 3, 10}, fl::dtype::f16), true);

  auto sample_mean = mean(input, {3});
  auto sample_var = var(input, {3}, true);
  auto true_out = (input - tileAs(sample_mean, input).astype(input.type())) /
      tileAs(fl::sqrt(sample_var + eps), input).astype(input.type());

  // no affine transform
  auto module1 = LayerNorm(feat_axes, eps, false);

  module1.train();
  auto out = module1.forward(input);

  ASSERT_TRUE(allClose(out, true_out.astype(out.type()), eps));

  module1.eval();
  out = module1.forward(input);

  ASSERT_TRUE(
      allClose(out.tensor(), true_out.tensor().astype(out.type()), eps));

  // with affine transform
  auto module2 = LayerNorm(feat_axes, eps, true);

  module2.train();
  auto out_train = module2.forward(input);
  module2.eval();
  auto out_eval = module2.forward(input);

  ASSERT_TRUE(allClose(out_train.tensor(), out_eval.tensor(), eps));
  ASSERT_EQ(out_train.shape(), input.shape());

  module2.train();
}
