/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <math.h>
#include <array>
#include <functional>
#include <stdexcept>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

using namespace fl;

namespace {

using JacobianFunc = std::function<Variable(Variable&)>;
bool jacobianTestImpl(
    const JacobianFunc& func,
    Variable& input,
    float precision = 1E-5,
    float perturbation = 1E-4) {
  auto fwdJacobian =
      Tensor({func(input).elements(), input.elements()}, fl::dtype::f32);

  for (int i = 0; i < input.elements(); ++i) {
    Tensor orig = input.tensor().flatten()(i);
    input.tensor().flat(i) = orig - perturbation;
    auto outa = func(input).tensor();

    input.tensor().flat(i) = orig + perturbation;
    auto outb = func(input).tensor();
    input.tensor().flat(i) = orig;

    fwdJacobian(fl::span, i) =
        fl::reshape((outb - outa), {static_cast<Dim>(outa.elements())}) * 0.5 /
        perturbation;
  }

  auto bwdJacobian =
      Tensor({func(input).elements(), input.elements()}, fl::dtype::f32);
  auto dout =
      Variable(fl::full(func(input).shape(), 0, func(input).type()), false);

  for (int i = 0; i < dout.elements(); ++i) {
    dout.tensor().flat(i) = 1; // element in 1D view
    input.zeroGrad();
    auto out = func(input);
    out.backward(dout);

    bwdJacobian(i) = fl::reshape(input.grad().tensor(), {input.elements()});
    dout.tensor().flat(i) = 0;
  }
  return allClose(fwdJacobian, bwdJacobian, precision);
}

class AutogradTestF16 : public ::testing::Test {
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

TEST(AutogradTest, Multiply) {
  auto x = Variable(fl::rand({5}), true);
  auto y = x * x;
  auto dy = Variable(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.tensor(), 2 * x.tensor()));
}

TEST(AutogradTest, BasicOps) {
  using FuncVar = std::function<Variable(Variable&, Variable&)>;
  using FuncScalarL = std::function<Variable(double, Variable&)>;
  using FuncScalarR = std::function<Variable(Variable&, double)>;
  auto testImpl = [](FuncVar fn1, FuncScalarL fn2, FuncScalarR fn3) {
    auto input = Variable(fl::rand({3, 4, 5, 6}, fl::dtype::f64) + 1, true);
    auto temp = Variable(fl::rand({3, 4, 5, 6}, fl::dtype::f64) - 2, false);
    JacobianFunc fnArrL = [&](Variable& in) { return fn1(in, temp); };
    ASSERT_TRUE(jacobianTestImpl(fnArrL, input));
    JacobianFunc fnArrR = [&](Variable& in) { return fn1(temp, in); };
    ASSERT_TRUE(jacobianTestImpl(fnArrR, input));
    JacobianFunc fnScalarL = [&](Variable& in) { return fn2(1.414, in); };
    ASSERT_TRUE(jacobianTestImpl(fnScalarL, input, 1E-5, 1E-7));
    JacobianFunc fnScalarR = [&](Variable& in) { return fn3(in, 1.732); };
    ASSERT_TRUE(jacobianTestImpl(fnScalarR, input, 1E-5, 1E-7));
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

TEST(AutogradTest, OperatorParenthesis) {
  auto x = Variable(fl::rand({1, 3, 3}, fl::dtype::f64), true);
  auto y = x(0, 0) + x(0, 1);
  auto funcOperatorParen = [](Variable& in) { return in(0, 0) + in(0, 1); };
  ASSERT_TRUE(jacobianTestImpl(funcOperatorParen, x));
}

TEST(AutogradTest, MultiplyAdd) {
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

TEST(AutogradTest, AutogradOperatorTypeCompatibility) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto f16 = Variable(fl::rand({2, 2}, fl::dtype::f16), true);
  auto f32 = Variable(fl::rand({2, 2}, fl::dtype::f32), true);

  // Binary operators
  EXPECT_THROW({ auto res = f16 + f32; }, std::invalid_argument); // +
  EXPECT_THROW({ auto res = f16 - f32; }, std::invalid_argument); // -
  EXPECT_THROW({ auto res = f16 * f32; }, std::invalid_argument); // *
  EXPECT_THROW({ auto res = f16 / f32; }, std::invalid_argument); // /
  EXPECT_THROW({ auto res = f16 > f32; }, std::invalid_argument); // >
  EXPECT_THROW({ auto res = f16 < f32; }, std::invalid_argument); // <
  EXPECT_THROW({ auto res = f16 >= f32; }, std::invalid_argument); // >=
  EXPECT_THROW({ auto res = f16 <= f32; }, std::invalid_argument); // <=
  EXPECT_THROW({ auto res = f16 && f32; }, std::invalid_argument); // &&
  EXPECT_THROW({ max(f16, f32); }, std::invalid_argument); // max
  EXPECT_THROW({ min(f16, f32); }, std::invalid_argument); // min
  EXPECT_THROW({ matmul(f16, f32); }, std::invalid_argument); // matmul
  EXPECT_THROW({ matmulTN(f16, f32); }, std::invalid_argument); // matmulTN
  EXPECT_THROW({ matmulNT(f16, f32); }, std::invalid_argument); // matmulNT
  EXPECT_NO_THROW({ binaryCrossEntropy(f16, f32); });
  EXPECT_NO_THROW({
    categoricalCrossEntropy(
        Variable(fl::rand({7, 10, 4}, fl::dtype::f16), true),
        Variable(
            (fl::rand({10, 4}, fl::dtype::u32) % 7).astype(fl::dtype::s32),
            false));
  });
  EXPECT_NO_THROW({ pool2d(f16, 1, 1, 1, 1, 1, 1); });
  EXPECT_NO_THROW({ embedding(f16, f32); }); // lookup is of a different type
  // Ternary operators
  auto f32_2 = Variable(fl::rand({2, 2}, fl::dtype::f32), true);
  auto f16_2 = Variable(fl::rand({2, 2}, fl::dtype::f16), true);
  EXPECT_THROW({ linear(f16, f32, f16_2); }, std::invalid_argument); // linear
  EXPECT_THROW({ linear(f16, f32, f32_2); }, std::invalid_argument); // linear
  auto w = Variable(fl::rand({1}, fl::dtype::f32), true);
  auto b = Variable(fl::rand({1}, fl::dtype::f32), true);
  EXPECT_THROW(
      { batchnorm(f16, f32, f32_2, w, b, {1}, true, 0.01, 0.01); },
      std::invalid_argument);
  EXPECT_THROW(
      { batchnorm(f16, f32, f16_2, w, b, {1}, true, 0.01, 0.01); },
      std::invalid_argument);
  EXPECT_THROW(
      { conv2d(f16, f32, f16_2, 1, 1, 0, 0, 1, 1); }, std::invalid_argument);
  // Quaternary
  auto f16_3 = Variable(fl::rand({2, 2, 3}, fl::dtype::f16), false);
  auto f16_4 = Variable(fl::rand({50}, fl::dtype::f16), false);
  EXPECT_THROW(
      {
        rnn(f16_3,
            Variable(Tensor(fl::dtype::f32), false),
            Variable(Tensor(fl::dtype::f32), false),
            f16_4,
            2,
            2,
            RnnMode::LSTM,
            true,
            0.0);
      },
      std::invalid_argument);
  // Variadic operators
  std::vector<Variable> concatInputs = {f16, f32, f16_2, f32_2};
  EXPECT_THROW({ concatenate(concatInputs, 0); }, std::invalid_argument);
}

TEST(AutogradTest, CastingAsDifferentGradTypes) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto f32 = Variable(fl::rand({5, 5}), true);
  auto f16 = Variable(fl::rand({5, 5}, fl::dtype::f16), true);
  // Computing gradients with mixed types fails when the op is applied
  ASSERT_THROW({ f32 + f16; }, std::invalid_argument);
}

TEST(AutogradTest, CastingAs) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto var = Variable(fl::rand({5, 5}), true);
  auto varF16 = var.astype(fl::dtype::f16);
  ASSERT_EQ(var.type(), fl::dtype::f32);
  ASSERT_EQ(varF16.type(), fl::dtype::f16);
  ASSERT_TRUE(allClose(varF16.tensor(), var.astype(fl::dtype::f16).tensor()));
}

TEST(AutogradTest, CastingAsBackward) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto a = Variable(fl::rand({4, 4}, fl::dtype::f16), true);
  auto b = Variable(fl::rand({4, 4}, fl::dtype::f16), false);
  auto c = b + a;
  c.backward();
  ASSERT_EQ(a.grad().type(), fl::dtype::f16);
  ASSERT_EQ(a.grad().type(), fl::dtype::f16);
  a = a.astype(fl::dtype::f32);
  ASSERT_FALSE(a.isGradAvailable());
}

TEST(AutogradTest, CastingAsGrad) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // compare to f32 case
  auto x = Variable(fl::full({5}, 2.0), true);
  auto y = Variable(fl::full({5}, 3.0), true);
  auto z = x * x + x * y + y * y;
  auto dz = Variable(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();

  // f16 -- cast gradients in both directions
  auto x32 = Variable(fl::full({5}, 2.0), true);
  auto y32 = Variable(fl::full({5}, 3.0), true);
  auto xf16 = x32.astype(fl::dtype::f16);
  auto yf16 = y32.astype(fl::dtype::f16);
  auto zf16 = xf16 * xf16 + xf16 * yf16 + yf16 * yf16;
  auto zf32 = zf16.astype(fl::dtype::f32);
  zf32.backward(dz);

  ASSERT_EQ(xf16.grad().type(), fl::dtype::f16);
  ASSERT_EQ(yf16.grad().type(), fl::dtype::f16);
  ASSERT_EQ(zf16.grad().type(), fl::dtype::f16);
  ASSERT_EQ(x32.grad().type(), fl::dtype::f32);
  ASSERT_EQ(y32.grad().type(), fl::dtype::f32);
  ASSERT_TRUE(
      allClose(dx.tensor(), xf16.grad().tensor().astype(fl::dtype::f32)));
  ASSERT_TRUE(
      allClose(dy.tensor(), y32.grad().tensor().astype(fl::dtype::f32)));
  ASSERT_TRUE(allClose(dx.tensor(), x32.grad().tensor()));
  ASSERT_TRUE(allClose(dy.tensor(), y32.grad().tensor()));
}

TEST(AutogradTest, NoCalcGrad) {
  auto x = Variable(fl::rand({5}), false);
  auto y = Variable(fl::rand({5}), true);
  auto z = x * x + x * y + y * y;
  auto dz = Variable(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dy.tensor(), 2 * y.tensor() + x.tensor()));
  ASSERT_THROW(x.grad(), std::logic_error);
}

TEST(AutogradTest, MultiplySub) {
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

TEST(AutogradTest, DivideAdd) {
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

TEST(AutogradTest, MultiplyAddScalar) {
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

TEST(AutogradTest, Exp) {
  auto x = Variable(fl::rand({5}), true);
  auto y = exp(x);
  auto dy = Variable(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.tensor(), (fl::exp(x.tensor()))));
}

TEST(AutogradTest, Pow) {
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

TEST(AutogradTest, Sigmoid) {
  auto x = Variable(fl::rand({5}), true);
  auto y = sigmoid(x);
  auto dy = Variable(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.tensor(), (y.tensor() * (1 - y.tensor()))));
  ASSERT_TRUE(allClose(
      dx.tensor(), (fl::sigmoid(x.tensor()) * (1 - fl::sigmoid(x.tensor())))));
}

TEST(AutogradTest, Erf) {
  auto x = Variable(fl::rand({5}), true);
  auto y = erf(x);
  ASSERT_TRUE(allClose(fl::erf(x.tensor()), y.tensor()));

  auto dy = Variable(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto targetGrads = 2 / std::sqrt(M_PI) * exp(negate(x * x));
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.tensor(), targetGrads.tensor()));

  auto funcErf = [](Variable& in) { return erf(in); };
  ASSERT_TRUE(jacobianTestImpl(funcErf, x, 5e-4, 1e-4));
}

TEST(AutogradTest, Tanh) {
  auto x = Variable(fl::rand({5}), true);
  auto y = tanh(x);
  auto dy = Variable(fl::full({5}, 1.0), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.tensor(), (1 - y.tensor() * y.tensor())));
  ASSERT_TRUE(allClose(
      dx.tensor(), (1 + fl::tanh(x.tensor())) * (1 - fl::tanh(x.tensor()))));
}

TEST(AutogradTest, Transpose) {
  auto in = Variable(fl::rand({5, 6, 7, 8}), true);
  auto out = transpose(in, {2, 0, 1, 3});
  out.backward();
  ASSERT_EQ(in.grad().shape(), Shape({5, 6, 7, 8}));

  auto funcErf = [](Variable& in) { return transpose(in, {1, 3, 2, 0}); };
  ASSERT_TRUE(jacobianTestImpl(funcErf, in, 5e-4, 1e-4));

  auto in2 = Variable(fl::rand({6, 7, 8, 9}), true);
  auto out2 = transpose(in2);
  out2.backward();
  ASSERT_EQ(in2.grad().shape(), Shape({6, 7, 8, 9}));

  auto funcErf2 = [](Variable& in) { return transpose(in); };
  ASSERT_TRUE(jacobianTestImpl(funcErf2, in2, 5e-4, 1e-4));
}

TEST(AutogradTest, Concatenate) {
  auto x1 = Variable(fl::rand({2, 3, 1, 2}, fl::dtype::f64), true);
  auto x2 = Variable(fl::rand({2, 3, 3, 2}, fl::dtype::f64), true);
  auto x3 = Variable(fl::rand({2, 3, 1, 2}, fl::dtype::f64), true);
  auto x4 = Variable(fl::rand({2, 3, 7, 2}, fl::dtype::f64), true);
  std::vector<Variable> inputs = {x1, x2, x3, x4};
  auto output = concatenate(inputs, 2);

  ASSERT_EQ(output.shape(), Shape({2, 3, 12, 2}));

  auto funcConcatenateT1 = [x2, x3, x4](Variable& in) {
    return concatenate({in, x2, x3, x4}, 2);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConcatenateT1, x1));

  auto funcConcatenateT2 = [x1, x2, x4](Variable& in) {
    return concatenate({x1, x2, in, x4}, 2);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConcatenateT2, x3));
}

TEST(AutogradTest, Split) {
  // check output
  auto x = Variable(fl::arange({7, 2}), true);
  auto yVec = split(x, 1, 0);
  ASSERT_EQ(yVec.size(), 7);
  ASSERT_EQ(yVec[0].shape(), Shape({1, 2}));
  ASSERT_EQ(yVec[2].shape(), Shape({1, 2}));
  ASSERT_TRUE(fl::all(yVec[6].tensor() == 6).scalar<char>());

  auto a = Variable(fl::arange({5, 3}, 1), true);
  auto bVec = split(a, {2, 1}, 1);
  ASSERT_EQ(bVec.size(), 2);
  ASSERT_EQ(bVec[0].shape(), Shape({5, 2}));
  ASSERT_EQ(bVec[1].shape(), Shape({5, 1}));
  ASSERT_TRUE(
      fl::all(bVec[0].tensor() == fl::arange({5, 2}, 1)).scalar<char>());
  ASSERT_TRUE(fl::all(bVec[1].tensor() == 2).scalar<char>());

  // check exception handling
  ASSERT_THROW(split(a, {2, 2}, 0), std::invalid_argument);

  // check gradient
  auto gradFunc = [](Variable& in) { return split(in, 2, 1)[0]; };
  auto input = Variable(fl::rand({2, 3}, fl::dtype::f64), true);
  ASSERT_TRUE(jacobianTestImpl(gradFunc, input));
}

TEST(AutogradTest, TileAs) {
  auto x = Variable(fl::rand({5}), true);
  auto y = Variable(fl::rand({5, 2}), true);
  auto z = y * tileAs(x, y);
  auto dz = Variable(fl::full({5, 2}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dy.tensor(), fl::tile(x.tensor(), {1, 2})));
  ASSERT_TRUE(allClose(dx.tensor(), fl::sum(y.tensor(), {1})));
}

TEST_F(AutogradTestF16, TileAsF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto x = Variable(fl::rand({5}, fl::dtype::f16), true);
  auto y = Variable(fl::rand({5, 2}, fl::dtype::f16), true);
  auto z = y * tileAs(x, y);
  ASSERT_EQ(x.type(), z.type());
  auto dz = Variable(fl::full({5, 2}, 1.0, fl::dtype::f16), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(
      dy.tensor(), fl::tile(x.tensor(), {1, 2}).astype(dx.type()), 1e-2));
  ASSERT_TRUE(
      allClose(dx.tensor(), fl::sum(y.tensor(), {1}).astype(dx.type()), 1e-2));
}

TEST(AutogradTest, TileAs2) {
  auto x = Variable(fl::rand({10}), true);
  auto z = tileAs(x, Shape({10, 3}));
  auto dz = Variable(fl::full({10, 3}, 1.0), false);
  z.backward(dz);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.tensor(), fl::full(x.shape(), 3.0)));
}

TEST(AutogradTest, Tile) {
  auto x = Variable(fl::rand({6}), true);
  auto y = Variable(fl::rand({6, 3}), true);
  auto z = y * tile(x, {1, 3});
  auto dz = Variable(fl::full({6, 3}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dy.tensor(), fl::tile(x.tensor(), {1, 3})));
  ASSERT_TRUE(allClose(dx.tensor(), fl::sum(y.tensor(), {1})));

  // Jacobian
  auto input = Variable(fl::rand({10, 1, 5}), true);
  auto funcTile = [](Variable& in) { return tile(in, {1, 2}); };
  ASSERT_TRUE(jacobianTestImpl(funcTile, input, 1E-4, 1E-3));
}

TEST(AutogradTest, Clamp) {
  auto input = Variable(fl::rand({5, 6, 7, 4}, fl::dtype::f64) * 3, true);
  double lo = -1.0, hi = 1.0;
  float perturb = 1E-5;
  // Need to do this as gradient is not continuous when input = lo / hi.
  auto& inarr = input.tensor();
  inarr = fl::where(fl::abs(inarr - lo) > perturb, inarr, lo + 10 * perturb);
  inarr = fl::where(fl::abs(inarr - hi) > perturb, inarr, hi + 10 * perturb);

  auto funcCol = [lo, hi](Variable& in) { return clamp(in, lo, hi); };

  ASSERT_TRUE(jacobianTestImpl(funcCol, input, 1E-10, perturb));
}

TEST(AutogradTest, SumAs) {
  auto x = Variable(fl::rand({5}), true);
  auto y = Variable(fl::rand({5, 2}), true);
  auto z = x * sumAs(y, x);
  auto dz = Variable(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dy.tensor(), fl::tile(x.tensor(), {1, 2})));
  ASSERT_TRUE(allClose(dx.tensor(), fl::sum(y.tensor(), {1})));
}

TEST(AutogradTest, SumAs2) {
  auto y = Variable(fl::rand({5, 2}), true);
  auto z = sumAs(y, {5});
  auto dz = Variable(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dy.tensor(), fl::full({5, 2}, 1.0)));
}

TEST(AutogradTest, Sum) {
  for (const bool keepDims : {false, true}) {
    Shape s = {6};
    if (keepDims) {
      s = {6, 1};
    }

    auto x = Variable(fl::rand(s), true);
    auto y = Variable(fl::rand({6, 3}), true);

    auto z = x * sum(y, {1}, keepDims);
    auto dz = Variable(fl::full(s, 1.0), false);
    z.backward(dz);

    auto dy = y.grad();
    auto dx = x.grad();
    ASSERT_TRUE(allClose(dy.tensor(), fl::tile(x.tensor(), {1, 3})));
    ASSERT_TRUE(allClose(dx.tensor(), fl::sum(y.tensor(), {1}, keepDims)));

    // Reduce over 1-dim input
    auto funcMean_0 = [keepDims](const Variable& in) {
      return sum(in, {0}, keepDims);
    };
    auto in = Variable(fl::rand({6}), true);
    ASSERT_TRUE(jacobianTestImpl(funcMean_0, in, 5E-3));
    // Reduce over scalar input
    auto inScalar = Variable(fl::fromScalar(3.14), true);
    ASSERT_TRUE(jacobianTestImpl(funcMean_0, inScalar, 5E-3));
  }

  auto r = Variable(fl::rand({5, 6, 7, 8}), true);
  auto rOut = sum(r, {1, 2});
  auto rOutTensor = fl::sum(r.tensor(), {1, 2});
  ASSERT_TRUE(allClose(rOut.tensor(), rOutTensor));
}

TEST(AutogradTest, Log1p) {
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

TEST(AutogradTest, Sqrt) {
  auto x = Variable(fl::rand({5, 3}, fl::dtype::f64), true);
  auto funcSqrt = [](Variable& in) { return fl::sqrt(in); };
  ASSERT_TRUE(jacobianTestImpl(funcSqrt, x, 1E-3));
}

TEST(AutogradTest, Mean) {
  for (const bool keepDims : {false, true}) {
    Shape xShape = keepDims ? Shape({5, 1, 1}) : Shape({5});
    auto x = Variable(fl::rand(xShape), true);
    auto y = Variable(fl::rand({5, 3, 2}), true);
    auto varOut = mean(y, {1, 2}, keepDims);
    auto z = x * mean(y, {1, 2}, keepDims);
    auto dz = Variable(fl::full(x.shape(), 1.0), false);
    z.backward(dz);
    auto dy = y.grad();
    auto dx = x.grad();
    ASSERT_TRUE(allClose(dy.tensor(), fl::tile(x.tensor(), {1, 3, 2}) / 6));
    ASSERT_TRUE(allClose(dx.tensor(), fl::mean(y.tensor(), {1, 2}, keepDims)));

    auto a = Variable(fl::rand({5, 3, 2}, fl::dtype::f64), true);
    auto funcMean = [keepDims](Variable& in) {
      return mean(in, {1, 2}, keepDims);
    };
    ASSERT_TRUE(jacobianTestImpl(funcMean, a, 1E-4));

    auto q = Variable(fl::rand({5, 6, 7, 8}), false);
    auto qOut = mean(q, {1, 2}, keepDims);
    auto qOutTensor = fl::mean(q.tensor(), {1, 2}, keepDims);
    ASSERT_TRUE(allClose(qOut.tensor(), qOutTensor));

    auto funcMean_0 = [keepDims](Variable& in) {
      return mean(in, {0}, keepDims);
    };
    // Reduce over 1-dim input
    auto in = Variable(fl::rand({6}), true);
    ASSERT_TRUE(jacobianTestImpl(funcMean_0, in, 5E-3));
    // Reduce over scalar input
    auto inScalar = Variable(fl::fromScalar(3.14), true);
    ASSERT_TRUE(jacobianTestImpl(funcMean_0, inScalar, 5E-3));
  }
}

TEST(AutogradTest, Variance) {
  std::vector<bool> biased = {true, false};
  for (auto b : biased) {
    for (const bool keepDims : {false, true}) {
      auto x = Variable(fl::rand({5, 6, 7, 8}, fl::dtype::f64), true);

      // TODO:{fl::Tensor} -- enforce AF versioning and remediate
      // Behavior of the bias parameter in af::var was changed in
      // https://git.io/Jv5gF and is different in ArrayFire v3.7. If isbiased is
      // true, sample variance rather than population variance is used. The
      // flashlight API implements the opposite behavior to be consistent with
      // other libraries.
      bool afVarBiasArg = !b;

      auto expectedVar = fl::var(x.tensor(), {1}, afVarBiasArg, keepDims);
      auto calculatedVar = var(x, {1}, b, keepDims);
      ASSERT_TRUE(allClose(calculatedVar.tensor(), expectedVar));

      auto funcVar = [b, keepDims](Variable& in) {
        return var(in, {1, 2}, b, keepDims);
      };
      ASSERT_TRUE(jacobianTestImpl(funcVar, x, 1E-5, 1E-5));
    }
  }
}

TEST(AutogradTest, Norm) {
  auto x = Variable(fl::rand({5, 3}, fl::dtype::f64), true);
  for (const bool keepDims : {false, true}) {
    auto funcNorm2 = [keepDims](Variable& in) {
      return norm(in, {1}, 2, keepDims);
    };
    ASSERT_TRUE(jacobianTestImpl(funcNorm2, x, 1E-4));
    auto funcNorm1 = [keepDims](Variable& in) {
      return norm(in, {1}, 1, keepDims);
    };
    ASSERT_TRUE(jacobianTestImpl(funcNorm1, x, 1E-4));
    auto funcNorm3 = [keepDims](Variable& in) {
      return norm(in, {1}, 3, keepDims);
    };
    ASSERT_TRUE(jacobianTestImpl(funcNorm3, x, 1E-4));
  }
}

TEST(AutogradTest, Normalize) {
  auto x = Variable(fl::rand({5, 3}, fl::dtype::f64), true);
  auto funcNormalize2 = [](Variable& in) { return normalize(in, {1}); };
  ASSERT_TRUE(jacobianTestImpl(funcNormalize2, x));
  auto ys = funcNormalize2(x);
  ASSERT_TRUE(allClose(
      fl::sum(ys.tensor() * ys.tensor(), {1}),
      fl::full({5}, 1, fl::dtype::f64)));
  auto yb = normalize(x, {1}, 2, 1);
  ASSERT_TRUE(fl::all(fl::sqrt(fl::sum(yb.tensor() * yb.tensor(), {1})) <= 1)
                  .scalar<char>());
}

TEST(AutogradTest, Indexing) {
  auto x = Variable(fl::rand({5, 6, 7, 4}, fl::dtype::f64), true);

  auto funcCol = [](Variable& input) { return input(fl::span, 4); };
  ASSERT_TRUE(jacobianTestImpl(funcCol, x));

  auto funcRow = [](Variable& input) { return input(4); };
  ASSERT_TRUE(jacobianTestImpl(funcRow, x));

  auto funcSlice = [](Variable& input) {
    return input(fl::span, fl::span, 4);
  };
  ASSERT_TRUE(jacobianTestImpl(funcSlice, x));

  auto funcCols = [](Variable& input) {
    return input(fl::span, fl::range(2, 5));
  };
  ASSERT_TRUE(jacobianTestImpl(funcCols, x));

  auto funcRows = [](Variable& input) { return input(fl::range(2, 5)); };
  ASSERT_TRUE(jacobianTestImpl(funcRows, x));

  auto funcSlices = [](Variable& input) {
    return input(fl::span, fl::span, fl::range(2, 5));
  };
  ASSERT_TRUE(jacobianTestImpl(funcSlices, x));
  auto funcFlat = [](Variable& input) {
    return input.flat(fl::range(4, 100));
  };
  ASSERT_TRUE(jacobianTestImpl(funcFlat, x));
}

TEST(AutogradTest, Convolve) {
  auto in = Variable(fl::rand({10, 9, 8, 7}, fl::dtype::f32), true);
  auto wt = Variable(fl::rand({4, 3, 8, 6}, fl::dtype::f32), true);
  auto bs = Variable(fl::rand({1, 1, 6, 1}, fl::dtype::f32), true);
  int px = 2, py = 1;
  int sx = 1, sy = 1;
  int dx = 1, dy = 1;
  auto benchmarks = std::make_shared<detail::ConvBenchmarks>();
  auto funcConvIn = [&](Variable& input) {
    return conv2d(
        input,
        wt,
        // bs,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1,
        benchmarks);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConvIn, in, 0.06));
  auto funcConvWt = [&](Variable& weight) {
    return conv2d(
        in,
        weight,
        // bs,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1,
        benchmarks);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConvWt, wt, 0.06));
  auto funcConvBs = [&](Variable& bias) {
    return conv2d(
        in,
        wt,
        bias,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1,
        benchmarks);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConvBs, bs, 0.03));
}

TEST_F(AutogradTestF16, ConvolveF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  const float scaleFactor = 10.0; // scale the input to prevent grad underflow
  auto in =
      Variable(fl::rand({3, 1, 2, 1}, fl::dtype::f16) * scaleFactor, true);
  auto wt = Variable(fl::rand({2, 1, 2, 1}, fl::dtype::f16), true);
  auto bs = Variable(fl::rand({1, 1, 1, 1}, fl::dtype::f16), true);
  int px = 1, py = 1;
  int sx = 1, sy = 1;
  int dx = 1, dy = 1;
  auto benchmarks = std::make_shared<detail::ConvBenchmarks>();
  auto funcConvIn = [&](Variable& input) {
    return conv2d(
        input,
        wt,
        bs,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1,
        benchmarks);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConvIn, in, 5e-1, 0.1));
  auto funcConvWt = [&](Variable& weight) {
    return conv2d(
        in,
        weight,
        bs,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1,
        benchmarks);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConvWt, wt, 5e-2, 0.1));
  auto funcConvBs = [&](Variable& bias) {
    return conv2d(
        in,
        wt,
        bias,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1,
        benchmarks);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConvBs, bs, 3e-2, 0.1));
}

TEST(AutogradTest, ConvolveFilterGroups) {
  int channel = 8;
  int groups = 2;
  // w x h x c x b
  auto in = Variable(fl::rand({10, 9, channel, 7}, fl::dtype::f32), true);
  // w x h x in x out
  auto wt =
      Variable(fl::rand({4, 3, channel / groups, 6}, fl::dtype::f32), true);
  auto bs = Variable(fl::rand({1, 1, 6, 1}, fl::dtype::f32), true);

  int px = 2, py = 1;
  int sx = 1, sy = 1;
  int dx = 1, dy = 1;
  auto funcConvIn = [&](Variable& input) {
    return conv2d(input, wt, bs, sx, sy, px, py, dx, dy, groups);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConvIn, in, 0.06));
  auto funcConvWt = [&](Variable& weight) {
    return conv2d(in, weight, bs, sx, sy, px, py, dx, dy, groups);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConvWt, wt, 0.05));
  auto funcConvBs = [&](Variable& bias) {
    return conv2d(in, wt, bias, sx, sy, px, py, dx, dy, groups);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConvBs, bs, 0.02));
}

TEST(AutogradTest, ConvolveDilation) {
  auto in = Variable(fl::rand({10, 9, 8, 7}, fl::dtype::f32), true);
  auto wt = Variable(fl::rand({4, 3, 8, 6}, fl::dtype::f32), true);
  auto bs = Variable(fl::rand({1, 1, 6, 1}, fl::dtype::f32), true);
  int px = 2, py = 1;
  int sx = 1, sy = 1;
  int dx = 2, dy = 1;
  auto funcConvIn = [&](Variable& input) {
    return conv2d(
        input,
        wt,
        bs,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConvIn, in, 0.06));
  auto funcConvWt = [&](Variable& weight) {
    return conv2d(
        in,
        weight,
        bs,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConvWt, wt, 0.05));
  auto funcConvBs = [&](Variable& bias) {
    return conv2d(
        in,
        wt,
        bias,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1);
  };
  ASSERT_TRUE(jacobianTestImpl(funcConvBs, bs, 0.02));
}

TEST(AutogradTest, Padding) {
  auto in = Variable(fl::rand({3, 3}, fl::dtype::f32), true);
  auto funcPad = [&](Variable& input) {
    return padding(input, {{1, 2}, {0, 1}}, -1);
  };
  ASSERT_TRUE(jacobianTestImpl(funcPad, in, 1E-3));
}

TEST(AutogradTest, Pooling) {
  auto in = Variable(fl::rand({3, 3, 1, 1}, fl::dtype::f32), true);
  auto funcPool = [&](Variable& input) { return pool2d(input, 2, 2, 1, 1); };
  ASSERT_TRUE(jacobianTestImpl(funcPool, in, 1E-3));
}

TEST_F(AutogradTestF16, PoolingF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  const float inputScale = 2.0; // scale the input to prevent grad underflow
  auto in = Variable(inputScale * fl::rand({3, 3, 1, 1}, fl::dtype::f16), true);
  auto funcPool = [&](Variable& input) { return pool2d(input, 2, 2, 1, 1); };
  ASSERT_TRUE(jacobianTestImpl(funcPool, in, 1e1, 1e-1)); // TODO: investigate
}

TEST(AutogradTest, Softmax) {
  auto in = Variable(fl::rand({3, 5, 1}, fl::dtype::f64), true);
  auto funcSm = [&](Variable& input) { return softmax(input, 0); };

  ASSERT_TRUE(jacobianTestImpl(funcSm, in, 1E-5));
}

TEST_F(AutogradTestF16, SoftmaxF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto in = Variable(fl::rand({3, 5, 1}, fl::dtype::f16), true);
  auto funcSm = [&](Variable& input) { return softmax(input, 0); };

  ASSERT_TRUE(jacobianTestImpl(funcSm, in, 1E-2, 1e-1));
}

TEST(AutogradTest, LogSoftmax) {
  auto in = Variable(fl::rand({3, 5, 1}, fl::dtype::f64), true);
  auto funcLsm = [&](Variable& input) { return logSoftmax(input, 0); };

  ASSERT_TRUE(jacobianTestImpl(funcLsm, in, 1E-5));
}

TEST_F(AutogradTestF16, LogSoftmaxF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto in = Variable(fl::rand({3, 5, 1}, fl::dtype::f16), true);
  auto funcLsm = [&](Variable& input) { return logSoftmax(input, 0); };

  ASSERT_TRUE(jacobianTestImpl(funcLsm, in, 1E-2, 1e-1));
}

TEST(AutogradTest, BinaryCrossEntropy) {
  auto x = Variable(fl::rand({10}), true);
  auto y = Variable(fl::rand({10}), true);
  auto loss = binaryCrossEntropy(x, y);

  // bce loss should be positive
  ASSERT_TRUE(fl::all(loss.tensor() > 0).scalar<char>());
}

TEST(AutogradTest, CrossEntropy) {
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
    ASSERT_TRUE(jacobianTestImpl(func, x, 1E-5));
    auto funcIgnore = [&](Variable& input) {
      return categoricalCrossEntropy(input, y, mode, ignoreIdx);
    };
    ASSERT_TRUE(jacobianTestImpl(funcIgnore, x, 1E-5));
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

TEST(AutogradTest, Reorder) {
  auto in = Variable(fl::rand({3, 1, 4, 1}, fl::dtype::f32) * 2, true);
  auto funcReorder = [&](Variable& input) {
    return reorder(input, {2, 0, 3, 1});
  };
  ASSERT_TRUE(jacobianTestImpl(funcReorder, in, 1E-3));
}

TEST(AutogradTest, matmul) {
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
    ASSERT_TRUE(jacobianTestImpl(funcMatmulLhs, a, 1E-6))
        << "matmul lhs gradient: lhs " << a.shape() << " rhs " << b.shape();
    auto funcMatmulRhs = [&](Variable& input) { return matmul(a, input); };
    ASSERT_TRUE(jacobianTestImpl(funcMatmulRhs, b, 1E-6))
        << "matmul rhs gradient: lhs " << a.shape() << " rhs " << b.shape();

    // matmulTN
    auto funcMatmulTNLhs = [&](Variable& input) { return matmulTN(input, b); };
    ASSERT_TRUE(jacobianTestImpl(funcMatmulTNLhs, aT, 1E-6))
        << "matmulTN lhs gradient: lhs " << a.shape() << " rhs " << b.shape();
    auto funcMatmulTNRhs = [&](Variable& input) { return matmulTN(aT, input); };
    ASSERT_TRUE(jacobianTestImpl(funcMatmulTNRhs, b, 1E-6))
        << "matmulTN rhs gradient: lhs " << a.shape() << " rhs " << b.shape();

    // matmulNT
    auto funcMatmulNTLhs = [&](Variable& input) { return matmulNT(input, bT); };
    ASSERT_TRUE(jacobianTestImpl(funcMatmulNTLhs, a, 1E-6))
        << "matmulTN lhs gradient: lhs " << a.shape() << " rhs " << b.shape();
    auto funcMatmulNTRhs = [&](Variable& input) { return matmulNT(a, input); };
    ASSERT_TRUE(jacobianTestImpl(funcMatmulNTRhs, bT, 1E-6))
        << "matmulTN rhs gradient: lhs " << a.shape() << " rhs " << b.shape();
  }
}

TEST(AutogradTest, Glu) {
  auto in = Variable(fl::rand({3, 4, 5}, fl::dtype::f64), true);
  auto funcGlu = [&](Variable& input) { return gatedlinearunit(input, 1); };
  ASSERT_TRUE(jacobianTestImpl(funcGlu, in, 1E-5));
}

TEST(AutogradTest, Linear) {
  std::vector<int> batchsizes = {1, 5};
  for (auto b : batchsizes) {
    auto in = Variable(fl::rand({3, 4, b}, fl::dtype::f64) * 2 - 1, true);
    auto wt = Variable(fl::rand({6, 3}, fl::dtype::f64) * 2 - 1, true);
    auto bs = Variable(fl::rand({6}, fl::dtype::f64) * 2 - 1, true);
    auto funcLinIn = [&](Variable& input) { return linear(input, wt, bs); };
    ASSERT_TRUE(jacobianTestImpl(funcLinIn, in, 1E-8));
    auto funcLinWt = [&](Variable& weight) { return linear(in, weight, bs); };
    ASSERT_TRUE(jacobianTestImpl(funcLinWt, wt, 1E-8));
    auto funcLinBs = [&](Variable& bias) { return linear(in, wt, bias); };
    ASSERT_TRUE(jacobianTestImpl(funcLinBs, bs, 1E-8));
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
    ASSERT_TRUE(jacobianTestImpl(funcLinIn, in, 5E-2, 5E-1));
    auto funcLinWt = [&](Variable& weight) { return linear(in, weight, bs); };
    ASSERT_TRUE(jacobianTestImpl(funcLinWt, wt, 5E-2, 5E-1));
    auto funcLinBs = [&](Variable& bias) { return linear(in, wt, bias); };
    ASSERT_TRUE(jacobianTestImpl(funcLinBs, bs, 5E-2, 5E-1));
  }
}

TEST(AutogradTest, WeightNormLinear) {
  auto v = Variable(fl::rand({3, 2}), true);
  auto normDim = {1};
  auto g = Variable(norm(v, normDim).tensor(), true);
  auto in = Variable(fl::rand({2, 3}, fl::dtype::f32), true);

  auto funcWeightNormIn = [&](Variable& input) {
    auto w = v * tileAs(g / norm(v, normDim), v);
    return matmul(w, input);
  };
  ASSERT_TRUE(jacobianTestImpl(funcWeightNormIn, in, 1E-3));

  auto funcWeightNormV = [&](Variable& input) {
    auto w = input * tileAs(g / norm(input, normDim), input);
    return matmul(w, in);
  };
  ASSERT_TRUE(jacobianTestImpl(funcWeightNormV, v, 1E-2));

  auto funcWeightNormG = [&](Variable& input) {
    auto w = v * tileAs(input / norm(v, normDim), v);
    return matmul(w, in);
  };
  ASSERT_TRUE(jacobianTestImpl(funcWeightNormG, g, 5E-3));
}

TEST(AutogradTest, WeightNormConv) {
  auto v = Variable(fl::rand({3, 3, 3, 8}), true);
  auto normDim = {0, 1, 2};
  auto g = Variable(
      norm(v, normDim, /* p = */ 2, /* keepDims = */ true).tensor(), true);
  auto in = Variable(fl::rand({7, 7, 3, 8}) * 2 - 2, true);

  auto funcWeightNormIn = [&](Variable& input) {
    auto w = v *
        tileAs(g / norm(v, normDim, /* p = */ 2, /* keepDims = */ true), v);
    return conv2d(
        input,
        w,
        /* sx */ 1,
        /* sy */ 1,
        /* px */ 0,
        /* py */ 0,
        /* dx */ 1,
        /* dy */ 1,
        /* groups */ 1);
  };
  ASSERT_TRUE(jacobianTestImpl(funcWeightNormIn, in, 3E-1));

  auto funcWeightNormV = [&](Variable& input) {
    auto w = input *
        tileAs(g / norm(input, normDim, /* p = */ 2, /* keepDims = */ true),
               input);
    return conv2d(
        in,
        w,
        /* sx */ 1,
        /* sy */ 1,
        /* px */ 0,
        /* py */ 0,
        /* dx */ 1,
        /* dy */ 1,
        /* groups */ 1);
  };
  ASSERT_TRUE(jacobianTestImpl(funcWeightNormV, v, 2E-1));

  auto funcWeightNormG = [&](Variable& input) {
    auto w = v *
        tileAs(input / norm(v, normDim, /* p = */ 2, /* keepDims = */ true),
               v);
    return conv2d(
        in,
        w,
        /* sx */ 1,
        /* sy */ 1,
        /* px */ 0,
        /* py */ 0,
        /* dx */ 1,
        /* dy */ 1,
        /* groups */ 1);
  };
  ASSERT_TRUE(jacobianTestImpl(funcWeightNormG, g, 2E-1));
}

void testRnnImpl(RnnMode mode, fl::dtype precision = fl::dtype::f64) {
  int numLayers = 2;
  int hiddenSize = 2;
  int inputSize = 2;
  int batchSize = 2;
  int seqLength = 3;
  bool bidirectional = true;
  float expectedPrecision = precision == fl::dtype::f16 ? 5E-2 : 1E-5;
  float perturbation = precision == fl::dtype::f16 ? 1E-1 : 1E-4;

  auto in =
      Variable(fl::rand({inputSize, batchSize, seqLength}, precision), true);
  size_t nParams;

  switch (mode) {
    case RnnMode::TANH:
      nParams = 56;
      break;
    case RnnMode::LSTM:
      nParams = 224;
      break;
    case RnnMode::GRU:
      nParams = 168;
      break;
    default:
      throw std::invalid_argument("invalid RNN mode for the test");
  }

  auto w =
      Variable(fl::rand({static_cast<long long>(nParams)}, precision), true);

  auto funcRnnIn = [&](Variable& input) -> Variable {
    return std::get<0>(
        rnn(input,
            Variable().astype(precision),
            Variable().astype(precision),
            w,
            hiddenSize,
            numLayers,
            mode,
            bidirectional,
            0.0));
  };
  ASSERT_TRUE(jacobianTestImpl(funcRnnIn, in, expectedPrecision, perturbation));

  auto funcRnnW = [&](Variable& weights) -> Variable {
    return std::get<0>(
        rnn(in,
            Variable().astype(precision),
            Variable().astype(precision),
            weights,
            hiddenSize,
            numLayers,
            mode,
            bidirectional,
            0.0));
  };
  ASSERT_TRUE(jacobianTestImpl(funcRnnW, w, expectedPrecision, perturbation));

  // We get the correct gradient for hx
  auto hx = Variable(
      fl::rand(
          {inputSize, batchSize, numLayers * (1 + bidirectional)},
          fl::dtype::f64),
      true);
  auto funcRnnHx = [&](Variable& hiddenState) -> Variable {
    return std::get<0>(
        rnn(in,
            hiddenState.astype(precision),
            Variable().astype(precision),
            w,
            hiddenSize,
            numLayers,
            mode,
            bidirectional,
            0.0));
  };
  ASSERT_TRUE(jacobianTestImpl(funcRnnHx, hx, expectedPrecision, perturbation));

  // We can compute the gradient w.r.t. hy
  auto funcRnnInDhy = [&](Variable& input) -> Variable {
    return std::get<1>(
        rnn(input,
            Variable().astype(precision),
            Variable().astype(precision),
            w,
            hiddenSize,
            numLayers,
            mode,
            bidirectional,
            0.0));
  };
  ASSERT_TRUE(
      jacobianTestImpl(funcRnnInDhy, in, expectedPrecision, perturbation));

  if (mode == RnnMode::LSTM) {
    // We get the correct gradient for cx
    auto cx = Variable(
        fl::rand(
            {inputSize, batchSize, numLayers * (1 + bidirectional)},
            fl::dtype::f64),
        true);
    auto funcRnnCx = [&](Variable& cellState) -> Variable {
      return std::get<0>(
          rnn(in,
              Variable().astype(precision),
              cellState.astype(precision),
              w,
              hiddenSize,
              numLayers,
              mode,
              bidirectional,
              0.0));
    };
    ASSERT_TRUE(
        jacobianTestImpl(funcRnnCx, cx, expectedPrecision, perturbation));

    // We can compute the gradient w.r.t. cy
    auto funcRnnInDcy = [&](Variable& input) -> Variable {
      return std::get<2>(
          rnn(input,
              Variable().astype(precision),
              Variable().astype(precision),
              w,
              hiddenSize,
              numLayers,
              mode,
              bidirectional,
              0.0));
    };
    ASSERT_TRUE(
        jacobianTestImpl(funcRnnInDcy, in, expectedPrecision, perturbation));
  }
}

TEST(AutogradTest, Rnn) {
  if (FL_BACKEND_CPU) {
    GTEST_SKIP() << "RNN gradient computation not yet supported on CPU";
  }

  testRnnImpl(RnnMode::TANH);
}

TEST_F(AutogradTestF16, RnnF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  testRnnImpl(RnnMode::TANH, fl::dtype::f16);
}

TEST(AutogradTest, Lstm) {
  if (FL_BACKEND_CPU) {
    GTEST_SKIP() << "RNN LSTM graident computation not yet supported on CPU";
  }

  testRnnImpl(RnnMode::LSTM);
}
TEST_F(AutogradTestF16, LstmF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  testRnnImpl(RnnMode::LSTM, fl::dtype::f16);
}

TEST(AutogradTest, Gru) {
  if (FL_BACKEND_CPU) {
    GTEST_SKIP() << "RNN GRU graident computation not yet supported on CPU";
  }
  testRnnImpl(RnnMode::GRU);
}

TEST_F(AutogradTestF16, GruF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  testRnnImpl(RnnMode::GRU, fl::dtype::f16);
}

TEST(AutogradTest, Embedding) {
  int nWords = 10;
  auto input =
      Variable((fl::rand({4, 2}) * nWords).astype(fl::dtype::f32), false);
  auto weights = Variable(fl::randn({4, nWords}, fl::dtype::f64), true);
  auto funcEmbed = [&](Variable& w) { return embedding(input, w); };
  ASSERT_TRUE(jacobianTestImpl(funcEmbed, weights, 1E-5));
}

TEST(AutogradTest, BatchNormEvalModeOutputSingleAxis) {
  int featDims = 3;
  std::vector<int> featAxes = {2};
  // input order: HWCN, following the docs
  auto input = Variable(fl::rand({13, 13, featDims, 16}), false);
  auto runningMean = Variable(fl::rand({featDims}, input.type()), false);
  auto runningVar = Variable(fl::rand({featDims}, input.type()), false);
  auto weight = Variable(fl::rand({featDims}, input.type()), false);
  auto bias = Variable(fl::rand({featDims}, input.type()), false);

  auto out = (batchnorm(
      input,
      weight,
      bias,
      runningMean,
      runningVar,
      featAxes,
      false,
      0.0,
      1E-5));
  for (int i = 0; i < featDims; ++i) {
    std::array<fl::Index, 4> sel = {fl::span, fl::span, i, fl::span};
    auto thisInput = input.tensor()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.tensor().flatten()(i).scalar<float>();
    auto thisVar = runningVar.tensor().flatten()(i).scalar<float>();
    auto thisWeight = weight.tensor().flatten()(i).scalar<float>();
    auto thisBias = bias.tensor().flatten()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / sqrt(thisVar + 1E-5);
    expectedOut = expectedOut * thisWeight + thisBias;
    ASSERT_TRUE(allClose(
        out.tensor()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1E-5));
  }

  // test on empty weigts and bias
  out = (batchnorm(
      input,
      Variable(),
      Variable(),
      runningMean,
      runningVar,
      featAxes,
      false,
      0.0,
      1E-5));
  for (int i = 0; i < featDims; ++i) {
    std::array<fl::Index, 4> sel = {fl::span, fl::span, i, fl::span};
    auto thisInput = input.tensor()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.tensor().flatten()(i).scalar<float>();
    auto thisVar = runningVar.tensor().flatten()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / sqrt(thisVar + 1E-5);
    ASSERT_TRUE(allClose(
        out.tensor()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1E-5));
  }
}

TEST(AutogradTest, BatchNormEvalModeOutputMultipleAxis) {
  // input order: HWCN, following the docs
  std::vector<int> featAxes = {0, 1, 2};
  auto input = Variable(fl::rand({13, 13, 4, 16}), false);

  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Variable(fl::rand({nfeatures}, input.type()), false);
  auto runningVar = Variable(fl::rand({nfeatures}, input.type()), false);
  auto weight = Variable(fl::rand({nfeatures}, input.type()), false);
  auto bias = Variable(fl::rand({nfeatures}, input.type()), false);

  auto out = (batchnorm(
      input,
      weight,
      bias,
      runningMean,
      runningVar,
      featAxes,
      false,
      0.0,
      1E-5));
  for (int i = 0; i < nfeatures; ++i) {
    std::array<fl::Index, 4> sel = {
        i % 13, (i / 13) % 13, (i / 13) / 13, fl::span};
    auto thisInput = input.tensor()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.tensor().flatten()(i).scalar<float>();
    auto thisVar = runningVar.tensor().flatten()(i).scalar<float>();
    auto thisWeight = weight.tensor().flatten()(i).scalar<float>();
    auto thisBias = bias.tensor().flatten()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / sqrt(thisVar + 1e-5);
    expectedOut = expectedOut * thisWeight + thisBias;

    ASSERT_TRUE(allClose(
        out.tensor()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1e-4));
  }

  // test on empty weigts and bias
  out = (batchnorm(
      input,
      Variable(),
      Variable(),
      runningMean,
      runningVar,
      featAxes,
      false,
      0.0,
      1E-5));
  for (int i = 0; i < nfeatures; ++i) {
    std::array<fl::Index, 4> sel = {
        i % 13, (i / 13) % 13, (i / 13) / 13, fl::span};
    auto thisInput = input.tensor()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.tensor().flatten()(i).scalar<float>();
    auto thisVar = runningVar.tensor().flatten()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / sqrt(thisVar + 1e-5);
    ASSERT_TRUE(allClose(
        out.tensor()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 5e-5));
  }
}

TEST(AutogradTest, BatchNormTrainModeOutputSingleAxis) {
  int numFeat = 3;
  std::vector<int> featAxes = {2};
  double epsilon = 1E-5;
  auto input = Variable(fl::rand({13, 13, numFeat, 8}), true);
  auto weight = Variable(fl::rand({numFeat}), true);
  auto bias = Variable(fl::rand({numFeat}), true);
  auto runningMean = Variable(fl::rand({numFeat}), false);
  auto runningVar = Variable(fl::rand({numFeat}), false);

  auto out = batchnorm(
      input,
      weight,
      bias,
      runningMean,
      runningVar,
      featAxes,
      true,
      0.0,
      epsilon);

  auto todim = Shape({1, 1, numFeat});
  std::vector<int> nrmAxes = {0, 1, 3};
  auto avg = moddims(mean(input, nrmAxes), todim);
  auto variance =
      moddims(var(input, nrmAxes, true /* population var */), todim);
  auto expectedOut = (input - tileAs(avg, input)) /
      fl::sqrt(tileAs(variance, input) + epsilon);
  expectedOut = expectedOut * tileAs(moddims(weight, todim), input) +
      tileAs(moddims(bias, todim), input);
  ASSERT_TRUE(allClose(out.tensor(), expectedOut.tensor(), 1e-5));
}

TEST(AutogradTest, BatchNormTrainModeOutputMultipleAxis) {
  std::vector<int> featAxes = {0, 1, 2};
  auto input = Variable(fl::rand({13, 13, 4, 8}), true);

  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto weight = Variable(fl::rand({nfeatures}), true);
  auto bias = Variable(fl::rand({nfeatures}), true);
  auto runningMean = Variable(fl::rand({nfeatures}), false);
  auto runningVar = Variable(fl::rand({nfeatures}), false);

  auto out = batchnorm(
      input, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5);

  auto todim = Shape({nfeatures});
  std::vector<int> nrmAxes = {3};
  auto avg = moddims(mean(input, nrmAxes), todim);
  auto variance = moddims(var(input, nrmAxes, true), todim);

  for (int i = 0; i < nfeatures; ++i) {
    std::array<fl::Index, 4> sel = {
        i % 13, (i / 13) % 13, (i / 13) / 13, fl::span};
    auto thisInput = input.tensor()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = avg.tensor().flatten()(i).scalar<float>();
    auto thisVar = variance.tensor().flatten()(i).scalar<float>();
    auto thisWeight = weight.tensor().flatten()(i).scalar<float>();
    auto thisBias = bias.tensor().flatten()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / sqrt(thisVar + 1e-5);
    expectedOut = expectedOut * thisWeight + thisBias;
    ASSERT_TRUE(allClose(
        out.tensor()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1e-5));
  }
}

TEST(AutogradTest, BatchNormJacobian) {
  // Jacobian Test with trainMode = true;

  int numFeat = 3;
  std::vector<int> featAxes = {2};
  auto input = Variable(fl::rand({8, 8, numFeat, 16}, fl::dtype::f32), true);
  auto runningMean = Variable(fl::rand({numFeat}, fl::dtype::f32), false);
  auto runningVar = Variable(fl::rand({numFeat}, fl::dtype::f32), false);
  auto weight = Variable(fl::rand({numFeat}, fl::dtype::f32), true);
  auto bias = Variable(fl::rand({numFeat}, fl::dtype::f32), true);

  auto funcBnIn = [&](Variable& in) {
    return (batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(funcBnIn, input, 1e-2, 1e-4));

  auto funcBnWt = [&](Variable& wt) {
    return (batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(funcBnWt, weight, 1e-2, 1e-4));

  auto funcBnBs = [&](Variable& bs) {
    return (batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(funcBnBs, bias, 1e-2, 1e-4));
}

TEST_F(AutogradTestF16, BatchNormJacobianF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // Jacobian Test with trainMode = true;

  int numFeat = 3;
  std::vector<int> featAxes = {2};
  auto input = Variable(fl::rand({8, 8, numFeat, 16}, fl::dtype::f16), true);
  auto runningMean = Variable(fl::rand({numFeat}, fl::dtype::f32), false);
  auto runningVar = Variable(fl::rand({numFeat}, fl::dtype::f32), false);
  auto weight = Variable(fl::rand({numFeat}, fl::dtype::f32), true);
  auto bias = Variable(fl::rand({numFeat}, fl::dtype::f32), true);

  // Use larger perturbations to ensure gradients don't underflow with fp16

  auto funcBnIn = [&](Variable& in) {
    return (batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(funcBnIn, input, 5e-2, 1e-1));

  auto funcBnWt = [&](Variable& wt) {
    return (batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(funcBnWt, weight, 5e-2, 1e-1));

  auto funcBnBs = [&](Variable& bs) {
    return (batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(funcBnBs, bias, 5e-2, 1e-1));
}

TEST(AutogradTest, BatchNormJacobianMultipleAxes) {
  // Jacobian Test with  trainMode = true;
  std::vector<int> featAxes = {0, 1, 2};
  auto input = Variable(fl::rand({8, 8, 3, 16}, fl::dtype::f32), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto runningVar = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto weight = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);
  auto bias = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);

  auto funcBnIn = [&](Variable& in) {
    return (batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(funcBnIn, input, 1e-2, 1e-3));

  auto funcBnWt = [&](Variable& wt) {
    return (batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(funcBnWt, weight, 1e-2, 1e-3));

  auto funcBnBs = [&](Variable& bs) {
    return (batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(funcBnBs, bias, 1e-2, 1e-3));
}

TEST_F(AutogradTestF16, BatchNormJacobianMultipleAxesF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // Jacobian Test with trainMode = true;
  std::vector<int> featAxes = {0, 1, 2};
  auto input = Variable(fl::rand({2, 2, 2, 1}, fl::dtype::f16), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto runningVar = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto weight = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);
  auto bias = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);

  // Use larger perturbations to ensure gradients don't underflow with fp16

  auto funcBnIn = [&](Variable& in) {
    return (batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(
      jacobianTestImpl(funcBnIn, input, 5e-2, 1e-1)); // TODO: investigate

  auto funcBnWt = [&](Variable& wt) {
    return (batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(funcBnWt, weight, 5e-2, 1e-1));

  auto funcBnBs = [&](Variable& bs) {
    return (batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(funcBnBs, bias, 5e-2, 1e-1));
}

TEST(AutogradTest, LayerNormJacobian) {
  std::vector<int> featAxes = {0, 1, 2, 3};
  auto input = Variable(fl::rand({7, 7, 3, 10}), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto runningVar = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto weight = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);
  auto bias = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);

  auto funcLnIn = [&](Variable& in) {
    return batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5);
  };

  ASSERT_TRUE(jacobianTestImpl(funcLnIn, input, 1e-2, 1e-4));
}

TEST_F(AutogradTestF16, LayerNormJacobianF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  std::vector<int> featAxes = {0, 1, 2, 3};
  const float inputScale = 4.0; // scale the input to prevent grad underflow
  auto input =
      Variable(inputScale * fl::rand({2, 2, 2, 4}, fl::dtype::f16), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dim(ax);
  }
  auto runningMean = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto runningVar = Variable(fl::rand({nfeatures}, fl::dtype::f32), false);
  auto weight = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);
  auto bias = Variable(fl::rand({nfeatures}, fl::dtype::f32), true);

  auto funcLnIn = [&](Variable& in) {
    return batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5);
  };

  ASSERT_TRUE(jacobianTestImpl(funcLnIn, input, 1e-4, 1e-2));
}

TEST(AutogradTest, GetAdvancedIndex) {
  // TODO: remove me
  if (!FL_BACKEND_CUDA) {
    GTEST_SKIP()
        << "Advanced indexing operator unsupported for non-CUDA backends";
  }
  std::vector<fl::dtype> validIndexTypes = {
      fl::dtype::s32, fl::dtype::s64, fl::dtype::u32, fl::dtype::u64};
  for (const auto& dtype : validIndexTypes) {
    auto x = Variable(fl::rand({20, 50, 40, 30}, fl::dtype::f32), true);
    Tensor a({6}, dtype);
    a(0) = 0;
    a(1) = 15;
    a(2) = 6;
    a(3) = 1;
    a(4) = 10;
    a(5) = 6;
    Tensor b({3}, dtype);
    b(0) = 5;
    b(1) = 11;
    b(2) = 19;
    auto x2 = x(a, b, fl::span, fl::range(0, 4));
    auto y = sum(x2 * x2, {0, 1, 2, 3}, /* keepDims = */ true);
    auto res = 2 * sum(x2, {0, 1, 2, 3}, /* keepDims = */ true).tensor();
    y.backward();
    auto grad = sum(x.grad(), {0, 1, 2, 3}, /* keepDims = */ true).tensor();
    ASSERT_TRUE(allClose(grad, res, 1e-3));
  }
}

TEST(AutogradTest, GetAdvancedIndexF16) {
  // TODO: remove me
  if (!FL_BACKEND_CUDA) {
    GTEST_SKIP()
        << "Advanced indexing operator unsupported for non-CUDA backends";
  }
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }
  std::vector<fl::dtype> validIndexTypes = {
      fl::dtype::s32, fl::dtype::s64, fl::dtype::u32, fl::dtype::u64};
  for (const auto& dtype : validIndexTypes) {
    auto x = Variable(fl::rand({20, 50, 40, 30}, fl::dtype::f16), true);
    Tensor a({6}, dtype);
    a(0) = 0;
    a(1) = 15;
    a(2) = 6;
    a(3) = 1;
    a(4) = 10;
    a(5) = 6;
    Tensor b({3}, dtype);
    b(0) = 5;
    b(1) = 11;
    b(2) = 19;
    auto x2 = x(a, b, fl::span, fl::range(0, 4));
    ASSERT_EQ(x2.type(), fl::dtype::f16);
    auto y = sum(x2 * x2, {0, 1, 2, 3}, /* keepDims = */ true);
    auto res = 2 * sum(x2, {0, 1, 2, 3}, /* keepDims = */ true).tensor();
    y.backward();
    ASSERT_EQ(x.grad().type(), fl::dtype::f16);
    auto grad = sum(x.grad(), {0, 1, 2, 3}, /* keepDims = */ true).tensor();
    ASSERT_TRUE(allClose(grad, res, 1e-3));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
