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

TEST(ModuleTest, ViewFwd) {
  auto module = View(Shape({-1, 0, 6}));
  auto input = Variable(Tensor({1, 2, 3, 4}), true);
  ASSERT_EQ(module(input).shape(), Shape({2, 2, 6}));
}

TEST(ModuleTest, PaddingFwd) {
  auto module = Padding({{1, 2}, {3, 4}}, -1);
  auto input = Variable(fl::rand({1, 2, 3, 4}, fl::dtype::f64), true);
  auto output = module(input);
  ASSERT_EQ(output.shape(), Shape({4, 9, 3, 4}));
  ASSERT_TRUE(allClose(input, output(fl::range(1, 2), fl::range(3, 5))));
  ASSERT_NEAR(
      fl::sum(input.tensor()).scalar<double>(),
      fl::sum(output.tensor()).scalar<double>() + 408,
      1E-5);
}

TEST(ModuleTest, NormalizeFwd) {
  auto input = Variable(fl::rand({10, 3}, fl::dtype::f64), true);
  auto module = Normalize({1}, 2, 1e-12, 5);
  module.train();
  auto out = module.forward(input);
  ASSERT_TRUE(allClose(
      fl::sqrt(fl::sum(out.tensor() * out.tensor(), {1})),
      fl::full({10}, 5, fl::dtype::f64)));
}

TEST(ModuleTest, TransformFwd) {
  auto inVar = Variable(fl::full({4, 5}, 1.0), true);

  auto l = Transform([](const Variable& in) { return fl::log(in); });

  ASSERT_TRUE(allClose(l.forward(inVar).tensor(), fl::full(inVar.shape(), 0.0)));
}

TEST(ModuleTest, PrecisionCastFwd) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half precision not available on this device";
  }

  auto in = Variable(fl::full({3, 3}, 1.0), true);
  auto precisionCast = PrecisionCast(fl::dtype::f16);
  auto out = precisionCast.forward(in);

  ASSERT_EQ(out.type(), fl::dtype::f16);
  ASSERT_TRUE(allClose(in.tensor(), out.astype(fl::dtype::f32).tensor()));
}

TEST(ModuleTest, IdentityFwd) {
  auto module = Identity();
  std::vector<Variable> in = {
      Variable(fl::rand({1000, 1000}), true),
      Variable(fl::rand({100, 100}), true)};

  // Train Mode
  module.train();
  auto out = module(in);
  ASSERT_EQ(out.size(), 2);
  ASSERT_TRUE(allClose(out.at(0), in.at(0), 1e-20));
  ASSERT_TRUE(allClose(out.at(1), in.at(1), 1e-20));

  // Eval Mode
  module.eval();
  out = module(in);
  ASSERT_EQ(out.size(), 2);
  ASSERT_TRUE(allClose(out.at(0), in.at(0), 1e-20));
  ASSERT_TRUE(allClose(out.at(1), in.at(1), 1e-20));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
