/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"

using namespace ::testing;
using namespace fl;

using fl::detail::AutogradTestF16;

TEST(AutogradUnaryOpsTest, Clamp) {
  auto input = Variable(fl::rand({5, 6, 7, 4}, fl::dtype::f64) * 3, true);
  double lo = -1.0, hi = 1.0;
  float perturb = 1E-5;
  // Need to do this as gradient is not continuous when input = lo / hi.
  auto& inarr = input.tensor();
  inarr = fl::where(fl::abs(inarr - lo) > perturb, inarr, lo + 10 * perturb);
  inarr = fl::where(fl::abs(inarr - hi) > perturb, inarr, hi + 10 * perturb);

  auto funcCol = [lo, hi](Variable& in) { return clamp(in, lo, hi); };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcCol, input, 1E-10, perturb));
}

TEST(AutogradUnaryOpsTest, Glu) {
  auto in = Variable(fl::rand({3, 4, 5}, fl::dtype::f64), true);
  auto funcGlu = [&](Variable& input) { return gatedlinearunit(input, 1); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcGlu, in, 1E-5));
}

TEST(AutogradUnaryOpsTest, Sigmoid) {
  auto x = Variable(fl::rand({5}), true);
  auto y = sigmoid(x);
  auto dy = Variable(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.tensor(), (y.tensor() * (1 - y.tensor()))));
  ASSERT_TRUE(allClose(
      dx.tensor(), (fl::sigmoid(x.tensor()) * (1 - fl::sigmoid(x.tensor())))));
}

TEST(AutogradUnaryOpsTest, Erf) {
  auto x = Variable(fl::rand({5}), true);
  auto y = erf(x);
  ASSERT_TRUE(allClose(fl::erf(x.tensor()), y.tensor()));

  auto dy = Variable(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto targetGrads = 2 / std::sqrt(M_PI) * exp(negate(x * x));
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.tensor(), targetGrads.tensor()));

  auto funcErf = [](Variable& in) { return erf(in); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcErf, x, 5e-4, 1e-4));
}

TEST(AutogradUnaryOpsTest, Tanh) {
  auto x = Variable(fl::rand({5}), true);
  auto y = tanh(x);
  auto dy = Variable(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.tensor(), (1 - y.tensor() * y.tensor())));
  ASSERT_TRUE(allClose(
      dx.tensor(), (1 + fl::tanh(x.tensor())) * (1 - fl::tanh(x.tensor()))));
}

TEST(AutogradUnaryOpsTest, Transpose) {
  auto in = Variable(fl::rand({5, 6, 7, 8}), true);
  auto out = transpose(in, {2, 0, 1, 3});
  out.backward();
  ASSERT_EQ(in.grad().shape(), Shape({5, 6, 7, 8}));

  auto funcErf = [](Variable& in) { return transpose(in, {1, 3, 2, 0}); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcErf, in, 5e-4, 1e-4));

  auto in2 = Variable(fl::rand({6, 7, 8, 9}), true);
  auto out2 = transpose(in2);
  out2.backward();
  ASSERT_EQ(in2.grad().shape(), Shape({6, 7, 8, 9}));

  auto funcErf2 = [](Variable& in) { return transpose(in); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcErf2, in2, 5e-4, 1e-4));
}

TEST(AutogradUnaryOpsTest, Exp) {
  auto x = Variable(fl::rand({5}), true);
  auto y = exp(x);
  auto dy = Variable(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.tensor(), (fl::exp(x.tensor()))));
}

TEST(AutogradUnaryOpsTest, Log1p) {
  auto x = Variable(fl::rand({5}), true);
  auto y = log1p(x);

  auto xCopy = Variable(x.tensor(), true);
  auto yExp = log(1 + xCopy);

  y.backward();
  yExp.backward();

  ASSERT_TRUE(allClose(y.tensor(), yExp.tensor()));
  ASSERT_TRUE(allClose(y.grad().tensor(), yExp.grad().tensor()));
  ASSERT_TRUE(allClose(x.grad().tensor(), xCopy.grad().tensor()));
}

TEST(AutogradUnaryOpsTest, Softmax) {
  auto in = Variable(fl::rand({3, 5, 1}, fl::dtype::f64), true);
  auto funcSm = [&](Variable& input) { return softmax(input, 0); };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcSm, in, 1E-5));
}

TEST_F(AutogradTestF16, SoftmaxF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto in = Variable(fl::rand({3, 5, 1}, fl::dtype::f16), true);
  auto funcSm = [&](Variable& input) { return softmax(input, 0); };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcSm, in, 1E-2, 1e-1));
}

TEST(AutogradUnaryOpsTest, LogSoftmax) {
  auto in = Variable(fl::rand({3, 5, 1}, fl::dtype::f64), true);
  auto funcLsm = [&](Variable& input) { return logSoftmax(input, 0); };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLsm, in, 1E-5));
}

TEST_F(AutogradTestF16, LogSoftmaxF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto in = Variable(fl::rand({3, 5, 1}, fl::dtype::f16), true);
  auto funcLsm = [&](Variable& input) { return logSoftmax(input, 0); };

  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLsm, in, 1E-2, 1e-1));
}

TEST(AutogradUnaryOpsTest, Pow) {
  {
    auto x = Variable(fl::rand({5}), true);
    auto y = pow(x, 2);
    auto dy = Variable(fl::full({5}, 2.0), false);
    y.backward(dy);
    auto dx = x.grad();
    ASSERT_TRUE(allClose(dx.tensor(), (2 * 2 * x.tensor())));
  }
  {
    auto x = Variable(fl::rand({5}), true);
    auto y = pow(x, 3);
    auto dy = Variable(fl::full({5}, 1.0), false);
    y.backward(dy);
    auto dx = x.grad();
    ASSERT_TRUE(allClose(dx.tensor(), (3 * fl::power(x.tensor(), 2))));
  }
}

TEST(AutogradUnaryOpsTest, Sqrt) {
  auto x = Variable(fl::rand({5, 3}, fl::dtype::f64), true);
  auto funcSqrt = [](Variable& in) { return fl::sqrt(in); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcSqrt, x, 1E-3));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
