/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"

using namespace ::testing;
using namespace fl;

using fl::detail::AutogradTestF16;

TEST(AutogradBinaryOpsTest, BasicOps) {
  using FuncVar = std::function<Variable(Variable&, Variable&)>;
  using FuncScalarL = std::function<Variable(double, Variable&)>;
  using FuncScalarR = std::function<Variable(Variable&, double)>;
  auto testImpl = [](FuncVar fn1, FuncScalarL fn2, FuncScalarR fn3) {
    auto input = Variable(fl::rand({3, 4, 5, 6}, fl::dtype::f64) + 1, true);
    auto temp = Variable(fl::rand({3, 4, 5, 6}, fl::dtype::f64) - 2, false);
    fl::detail::JacobianFunc fnArrL = [&](Variable& in) { return fn1(in, temp); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(fnArrL, input));
    fl::detail::JacobianFunc fnArrR = [&](Variable& in) { return fn1(temp, in); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(fnArrR, input));
    fl::detail::JacobianFunc fnScalarL = [&](Variable& in) { return fn2(1.414, in); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(fnScalarL, input, 1E-5, 1E-7));
    fl::detail::JacobianFunc fnScalarR = [&](Variable& in) { return fn3(in, 1.732); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(fnScalarR, input, 1E-5, 1E-7));
  };

  FuncVar funcAdd1 = [](Variable& a, Variable& b) { return a + b; };
  FuncScalarL funcAdd2 = [](double a, Variable& b) { return a + b; };
  FuncScalarR funcAdd3 = [](Variable& a, double b) { return a + b; };
  testImpl(funcAdd1, funcAdd2, funcAdd3);

  FuncVar funcSub1 = [](Variable& a, Variable& b) { return a - b; };
  FuncScalarL funcSub2 = [](double a, Variable& b) { return a - b; };
  FuncScalarR funcSub3 = [](Variable& a, double b) { return a - b; };
  testImpl(funcSub1, funcSub2, funcSub3);

  FuncVar funcDiv1 = [](Variable& a, Variable& b) { return a / b; };
  FuncScalarL funcDiv2 = [](double a, Variable& b) { return a / b; };
  FuncScalarR funcDiv3 = [](Variable& a, double b) { return a / b; };
  testImpl(funcDiv1, funcDiv2, funcDiv3);

  FuncVar funcMul1 = [](Variable& a, Variable& b) { return a * b; };
  FuncScalarL funcMul2 = [](double a, Variable& b) { return a * b; };
  FuncScalarR funcMul3 = [](Variable& a, double b) { return a * b; };
  testImpl(funcMul1, funcMul2, funcMul3);

  FuncVar funcMin1 = [](Variable& a, Variable& b) { return min(a, b); };
  FuncScalarL funcMin2 = [](double a, Variable& b) { return min(a, b); };
  FuncScalarR funcMin3 = [](Variable& a, double b) { return min(a, b); };
  testImpl(funcMin1, funcMin2, funcMin3);

  FuncVar funcMax1 = [](Variable& a, Variable& b) { return max(a, b); };
  FuncScalarL funcMax2 = [](double a, Variable& b) { return max(a, b); };
  FuncScalarR funcMax3 = [](Variable& a, double b) { return max(a, b); };
  testImpl(funcMax1, funcMax2, funcMax3);
}

TEST(AutogradBinaryOpsTest, BinaryCrossEntropy) {
  auto x = Variable(fl::rand({10}), true);
  auto y = Variable(fl::rand({10}), true);
  auto loss = binaryCrossEntropy(x, y);

  // bce loss should be positive
  ASSERT_TRUE(fl::all(loss.tensor() > 0).scalar<char>());
}

TEST(AutogradBinaryOpsTest, CrossEntropy) {
  auto x = Variable(fl::rand({7, 10, 4}, fl::dtype::f64), true);
  auto y = Variable(
      (fl::rand({10, 4}, fl::dtype::u32) % 7).astype(fl::dtype::s32), false);
  auto ignoreIdx = y(0, 0).scalar<int>();

  std::vector<ReduceMode> modes = {
      ReduceMode::NONE, ReduceMode::SUM, ReduceMode::MEAN};
  for (auto mode : modes) {
    auto func = [&](Variable& input) {
      return categoricalCrossEntropy(input, y, mode);
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(func, x, 1E-5));
    auto funcIgnore = [&](Variable& input) {
      return categoricalCrossEntropy(input, y, mode, ignoreIdx);
    };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcIgnore, x, 1E-5));
  }

  auto lossSum = categoricalCrossEntropy(x, y, ReduceMode::SUM);
  auto lossMean = categoricalCrossEntropy(x, y, ReduceMode::MEAN);
  ASSERT_NEAR((lossSum / lossMean).scalar<double>(), 40, 1e-5);

  auto lossSumIgnore =
      categoricalCrossEntropy(x, y, ReduceMode::SUM, ignoreIdx);
  auto lossMeanIgnore =
      categoricalCrossEntropy(x, y, ReduceMode::MEAN, ignoreIdx);
  auto ignoreCount = fl::sum(y.tensor() == ignoreIdx).scalar<unsigned>();
  ASSERT_NEAR(
      (lossSumIgnore / lossMeanIgnore).scalar<double>(),
      40 - ignoreCount,
      1e-5);

  ASSERT_THROW(
      categoricalCrossEntropy(
          Variable(fl::rand({4, 5, 6}), false),
          Variable(fl::rand({5, 8}), false)),
      std::invalid_argument);

  ASSERT_THROW(
      categoricalCrossEntropy(
          Variable(fl::rand({4, 5, 6}), false), Variable(fl::rand({5}), false)),
      std::invalid_argument);
}

TEST(AutogradBinaryOpsTest, Linear) {
  std::vector<int> batchsizes = {1, 5};
  for (auto b : batchsizes) {
    auto in = Variable(fl::rand({3, 4, b}, fl::dtype::f64) * 2 - 1, true);
    auto wt = Variable(fl::rand({6, 3}, fl::dtype::f64) * 2 - 1, true);
    auto bs = Variable(fl::rand({6}, fl::dtype::f64) * 2 - 1, true);
    auto funcLinIn = [&](Variable& input) { return linear(input, wt, bs); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLinIn, in, 1E-8));
    auto funcLinWt = [&](Variable& weight) { return linear(in, weight, bs); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLinWt, wt, 1E-8));
    auto funcLinBs = [&](Variable& bias) { return linear(in, wt, bias); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLinBs, bs, 1E-8));
  }
}

TEST_F(AutogradTestF16, LinearF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  std::vector<int> batchsizes = {1, 5};
  const float scale = 4.0; // scale prevent grad underflow
  for (auto b : batchsizes) {
    auto in = Variable(fl::rand({2, 2, b}, fl::dtype::f16) * scale, true);
    auto wt = Variable(fl::rand({2, 2}, fl::dtype::f16) * scale, true);
    auto bs = Variable(fl::rand({2}, fl::dtype::f16) * scale, true);
    auto funcLinIn = [&](Variable& input) { return linear(input, wt, bs); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLinIn, in, 5E-2, 5E-1));
    auto funcLinWt = [&](Variable& weight) { return linear(in, weight, bs); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLinWt, wt, 5E-2, 5E-1));
    auto funcLinBs = [&](Variable& bias) { return linear(in, wt, bias); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcLinBs, bs, 5E-2, 5E-1));
  }
}

TEST(AutogradBinaryOpsTest, Multiply) {
  auto x = Variable(fl::rand({5}), true);
  auto y = x * x;
  auto dy = Variable(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.tensor(), 2 * x.tensor()));
}

TEST(AutogradBinaryOpsTest, MultiplyAdd) {
  auto x = Variable(fl::rand({5}), true);
  auto y = Variable(fl::rand({5}), true);
  auto z = x * x + x * y + y * y;
  auto dz = Variable(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dx.tensor(), 2 * x.tensor() + y.tensor()));
  ASSERT_TRUE(allClose(dy.tensor(), 2 * y.tensor() + x.tensor()));
}

TEST(AutogradBinaryOpsTest, MultiplyAddScalar) {
  auto x = Variable(fl::rand({5}), true);
  auto y = Variable(fl::rand({5}), true);
  auto z = 2 * x + x * y + y;
  auto dz = Variable(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dx.tensor(), (2.0 + y.tensor())));
  ASSERT_TRUE(allClose(dy.tensor(), (1.0 + x.tensor())));
}

TEST(AutogradBinaryOpsTest, MultiplySub) {
  auto x = Variable(fl::rand({5}), true);
  auto y = Variable(fl::rand({5}), true);
  auto z = x * x - x * y;
  auto dz = Variable(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dx.tensor(), (2 * x.tensor() - y.tensor())));
  ASSERT_TRUE(allClose(dy.tensor(), (-x.tensor())));
}

TEST(AutogradBinaryOpsTest, DivideAdd) {
  auto x = Variable(fl::rand({5}, fl::dtype::f64), true);
  auto y = Variable(fl::rand({5}, fl::dtype::f64), true);
  auto z = x + x / y + y;
  auto dz = Variable(fl::full({5}, 1.0, fl::dtype::f64), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();
  ASSERT_EQ(z.type(), fl::dtype::f64);
  ASSERT_TRUE(allClose(dx.tensor(), (1.0 + 1.0 / y.tensor())));
  ASSERT_TRUE(
      allClose(dy.tensor(), (1.0 - x.tensor() / (y.tensor() * y.tensor()))));
}

TEST(AutogradBinaryOpsTest, matmul) {
  unsigned M = 10;
  unsigned K = 12;
  unsigned N = 14;
  unsigned b2 = 2;
  unsigned b3 = 4;
  auto mk = Shape({M, K});
  auto mkb2 = Shape({M, K, b2}); // 1 batch dim
  auto mkb2b3 = Shape({M, K, b2, b3}); // 2 batch dims
  auto kn = Shape({K, N});
  auto knb2 = Shape({K, N, b2}); // 1 batch dim
  auto knb2b3 = Shape({K, N, b2, b3}); // 2 batch dims

  // lhs, rhs
  std::vector<std::pair<Shape, Shape>> inputs = {
      {mk, kn},
      {mk, knb2},
      {mk, knb2b3},
      {mkb2, kn},
      {mkb2, knb2},
      {mkb2b3, kn},
      {mkb2b3, knb2b3}};

  auto trFirstTwoDims = [](const Shape& in) -> Shape {
    Shape out = in;
    auto out1 = out[1];
    out[1] = out[0];
    out[0] = out1;
    return out;
  };

  for (auto& pair : inputs) {
    auto& aShape = pair.first;
    auto& bShape = pair.second;

    auto a = Variable(fl::rand(aShape, fl::dtype::f64) * 2 - 1, true);
    auto b = Variable(fl::rand(bShape, fl::dtype::f64) * 2 - 1, true);

    auto aT = Variable(fl::rand(trFirstTwoDims(aShape), fl::dtype::f64), true);
    auto bT = Variable(fl::rand(trFirstTwoDims(bShape), fl::dtype::f64), true);

    // matmul
    auto funcMatmulLhs = [&](Variable& input) { return matmul(input, b); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMatmulLhs, a, 1E-6))
        << "matmul lhs gradient: lhs " << a.shape() << " rhs " << b.shape();
    auto funcMatmulRhs = [&](Variable& input) { return matmul(a, input); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMatmulRhs, b, 1E-6))
        << "matmul rhs gradient: lhs " << a.shape() << " rhs " << b.shape();

    // matmulTN
    auto funcMatmulTNLhs = [&](Variable& input) { return matmulTN(input, b); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMatmulTNLhs, aT, 1E-6))
        << "matmulTN lhs gradient: lhs " << a.shape() << " rhs " << b.shape();
    auto funcMatmulTNRhs = [&](Variable& input) { return matmulTN(aT, input); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMatmulTNRhs, b, 1E-6))
        << "matmulTN rhs gradient: lhs " << a.shape() << " rhs " << b.shape();

    // matmulNT
    auto funcMatmulNTLhs = [&](Variable& input) { return matmulNT(input, bT); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMatmulNTLhs, a, 1E-6))
        << "matmulTN lhs gradient: lhs " << a.shape() << " rhs " << b.shape();
    auto funcMatmulNTRhs = [&](Variable& input) { return matmulNT(a, input); };
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcMatmulNTRhs, bT, 1E-6))
        << "matmulTN rhs gradient: lhs " << a.shape() << " rhs " << b.shape();
  }
}

TEST(AutogradNormalizationTest, WeightNormLinear) {
  auto v = Variable(fl::rand({3, 2}), true);
  auto normDim = {1};
  auto g = Variable(norm(v, normDim).tensor(), true);
  auto in = Variable(fl::rand({2, 3}, fl::dtype::f32), true);

  auto funcWeightNormIn = [&](Variable& input) {
    auto w = v * tileAs(g / norm(v, normDim), v);
    return matmul(w, input);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcWeightNormIn, in, 1E-3));

  auto funcWeightNormV = [&](Variable& input) {
    auto w = input * tileAs(g / norm(input, normDim), input);
    return matmul(w, in);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcWeightNormV, v, 1E-2));

  auto funcWeightNormG = [&](Variable& input) {
    auto w = v * tileAs(input / norm(v, normDim), v);
    return matmul(w, in);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcWeightNormG, g, 5E-3));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
