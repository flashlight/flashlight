/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <array>
#include <functional>
#include <stdexcept>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/Init.h"
#include "flashlight/fl/common/common.h"

using namespace fl;

namespace {

using JacobianFunc = std::function<Variable(Variable&)>;
bool jacobianTestImpl(
    const JacobianFunc& func,
    Variable& input,
    float precision = 1E-5,
    float perturbation = 1E-4) {
  auto fwdJacobian =
      af::array(func(input).elements(), input.elements(), af::dtype::f32);

  for (int i = 0; i < input.elements(); ++i) {
    af::array orig = input.array()(i);
    input.array()(i) = orig - perturbation;
    auto outa = func(input).array();

    input.array()(i) = orig + perturbation;
    auto outb = func(input).array();
    input.array()(i) = orig;

    fwdJacobian(af::span, i) =
        af::moddims((outb - outa), outa.elements()) * 0.5 / perturbation;
  }

  auto bwdJacobian =
      af::array(func(input).elements(), input.elements(), af::dtype::f32);
  auto dout =
      Variable(af::constant(0, func(input).dims(), func(input).type()), false);
  for (int i = 0; i < dout.elements(); ++i) {
    dout.array()(i) = 1; // element in 1D view
    input.zeroGrad();
    auto out = func(input);
    out.backward(dout);
    bwdJacobian(i, af::span) =
        af::moddims(input.grad().array(), input.elements());
    dout.array()(i) = 0;
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

TEST(AutogradTest, AfRefCountBasic) {
  int refCount = 0;
  // Baseline af refcount behavior
  auto a = af::constant(1, {2, 2});
  af_get_data_ref_count(&refCount, a.get());
  ASSERT_EQ(refCount, 1);
  auto b = a;
  af_get_data_ref_count(&refCount, a.get());
  ASSERT_EQ(refCount, 2);

  // Behavior when wrapped by a variable
  auto v = Variable(af::constant(1, {2, 2}), false);
  auto w = v;
  // Should still be 1
  af_get_data_ref_count(&refCount, v.array().get());
  ASSERT_EQ(refCount, 1);
  af_get_data_ref_count(&refCount, w.array().get());
  ASSERT_EQ(refCount, 1);
}

TEST(AutogradTest, AfRefCountModify) {
  int refCount = 0;
  // Compositional operations don't increment refcount
  auto a = af::constant(1, {2, 2});
  auto b = af::constant(1, {2, 2});
  auto arrRes = a + b;
  af_get_data_ref_count(&refCount, a.get());
  ASSERT_EQ(refCount, 1);
  af_get_data_ref_count(&refCount, b.get());
  ASSERT_EQ(refCount, 1);
  // Multiple uses of the same variable doesn't push count
  auto c = af::constant(1, {2, 2});
  auto d = af::constant(1, {2, 2});
  auto arrResMult = c * c + d * d;
  af_get_data_ref_count(&refCount, c.get());
  ASSERT_EQ(refCount, 1);
  af_get_data_ref_count(&refCount, d.get());
  ASSERT_EQ(refCount, 1);

  // // Same behavior with Variables
  auto v = Variable(af::constant(1, {2, 2}), false);
  auto w = Variable(af::constant(1, {2, 2}), false);
  auto varRes = v + w;
  af_get_data_ref_count(&refCount, v.array().get());
  ASSERT_EQ(refCount, 1);
  af_get_data_ref_count(&refCount, w.array().get());
  ASSERT_EQ(refCount, 1);
  // Multiuse with variables
  auto y = Variable(af::constant(1, {2, 2}), false);
  auto z = Variable(af::constant(1, {2, 2}), false);
  auto varResMult = y * y + z * z;
  af_get_data_ref_count(&refCount, y.array().get());
  ASSERT_EQ(refCount, 1);
  af_get_data_ref_count(&refCount, z.array().get());
  ASSERT_EQ(refCount, 1);
}

TEST(AutogradTest, AfRefCountGradient) {
  int refCount = 0;
  // backward should never increase refcount of the underlying Array
  // It only takes a variable, so the user can't accidentally copy
  auto v = Variable(af::constant(1, {2, 2}), true);
  auto w = Variable(af::constant(1, {2, 2}), false);
  v.backward(w);
  af_get_data_ref_count(&refCount, v.array().get());
  ASSERT_EQ(refCount, 1);
  af_get_data_ref_count(&refCount, w.array().get());
  ASSERT_EQ(refCount, 1);

  // addGrad specifically is idempotent to refcount
  auto x = Variable(af::constant(1, {2, 2}), false);
  auto y = Variable(af::constant(1, {2, 2}), false);
  x.addGrad(y);
  af_get_data_ref_count(&refCount, x.array().get());
  ASSERT_EQ(refCount, 1);
  af_get_data_ref_count(&refCount, y.array().get());
  ASSERT_EQ(refCount, 1);
}

TEST(AutogradTest, AfArrBackwardNoMutate) {
  // backward preserves the gradient of v
  auto v = Variable(af::constant(1, {2, 2}), true);
  auto w = Variable(af::constant(1, {2, 2}), false);
  v.backward(w);
  ASSERT_TRUE(allClose(v.grad().array(), v.array()));
  // mutating doesn't change v's gradient
  w = w + 2;
  ASSERT_TRUE(allClose(v.grad().array(), v.array()));
}

TEST(AutogradTest, Multiply) {
  auto x = Variable(af::randu(5), true);
  auto y = x * x;
  auto dy = Variable(af::constant(1.0, 5), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.array(), 2 * x.array()));
}

TEST(AutogradTest, BasicOps) {
  using FuncVar = std::function<Variable(Variable&, Variable&)>;
  using FuncScalarL = std::function<Variable(double, Variable&)>;
  using FuncScalarR = std::function<Variable(Variable&, double)>;
  auto test_impl = [](FuncVar fn1, FuncScalarL fn2, FuncScalarR fn3) {
    auto input = Variable(af::randu(3, 4, 5, 6, af::dtype::f64) + 1, true);
    auto temp = Variable(af::randu(3, 4, 5, 6, af::dtype::f64) - 2, false);
    JacobianFunc fn_arr_l = [&](Variable& in) { return fn1(in, temp); };
    ASSERT_TRUE(jacobianTestImpl(fn_arr_l, input));
    JacobianFunc fn_arr_r = [&](Variable& in) { return fn1(temp, in); };
    ASSERT_TRUE(jacobianTestImpl(fn_arr_r, input));
    JacobianFunc fn_scalar_l = [&](Variable& in) { return fn2(1.414, in); };
    ASSERT_TRUE(jacobianTestImpl(fn_scalar_l, input, 1E-5, 1E-7));
    JacobianFunc fn_scalar_r = [&](Variable& in) { return fn3(in, 1.732); };
    ASSERT_TRUE(jacobianTestImpl(fn_scalar_r, input, 1E-5, 1E-7));
  };

  FuncVar func_add1 = [](Variable& a, Variable& b) { return a + b; };
  FuncScalarL func_add2 = [](double a, Variable& b) { return a + b; };
  FuncScalarR func_add3 = [](Variable& a, double b) { return a + b; };
  test_impl(func_add1, func_add2, func_add3);

  FuncVar func_sub1 = [](Variable& a, Variable& b) { return a - b; };
  FuncScalarL func_sub2 = [](double a, Variable& b) { return a - b; };
  FuncScalarR func_sub3 = [](Variable& a, double b) { return a - b; };
  test_impl(func_sub1, func_sub2, func_sub3);

  FuncVar func_div1 = [](Variable& a, Variable& b) { return a / b; };
  FuncScalarL func_div2 = [](double a, Variable& b) { return a / b; };
  FuncScalarR func_div3 = [](Variable& a, double b) { return a / b; };
  test_impl(func_div1, func_div2, func_div3);

  FuncVar func_mul1 = [](Variable& a, Variable& b) { return a * b; };
  FuncScalarL func_mul2 = [](double a, Variable& b) { return a * b; };
  FuncScalarR func_mul3 = [](Variable& a, double b) { return a * b; };
  test_impl(func_mul1, func_mul2, func_mul3);

  FuncVar func_min1 = [](Variable& a, Variable& b) { return min(a, b); };
  FuncScalarL func_min2 = [](double a, Variable& b) { return min(a, b); };
  FuncScalarR func_min3 = [](Variable& a, double b) { return min(a, b); };
  test_impl(func_min1, func_min2, func_min3);

  FuncVar func_max1 = [](Variable& a, Variable& b) { return max(a, b); };
  FuncScalarL func_max2 = [](double a, Variable& b) { return max(a, b); };
  FuncScalarR func_max3 = [](Variable& a, double b) { return max(a, b); };
  test_impl(func_max1, func_max2, func_max3);
}

TEST(AutogradTest, OperatorParenthesis) {
  auto x = Variable(af::randn(1, 3, 3, f64), true);
  auto y = x(0, 0) + x(0, 1);
  auto func_operator_paren = [](Variable& in) { return in(0, 0) + in(0, 1); };
  ASSERT_TRUE(jacobianTestImpl(func_operator_paren, x));
}

TEST(AutogradTest, MultiplyAdd) {
  auto x = Variable(af::randu(5), true);
  auto y = Variable(af::randu(5), true);
  auto z = x * x + x * y + y * y;
  auto dz = Variable(af::constant(1.0, 5), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dx.array(), 2 * x.array() + y.array()));
  ASSERT_TRUE(allClose(dy.array(), 2 * y.array() + x.array()));
}

TEST(AutogradTest, AutogradOperatorTypeCompatibility) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto f16 = Variable(af::randu({2, 2}, af::dtype::f16), true);
  auto f32 = Variable(af::randu({2, 2}, af::dtype::f32), true);

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
        Variable(af::randu(7, 10, 4, af::dtype::f16), true),
        Variable((af::randu(10, 4, af::dtype::u32) % 7).as(s32), false));
  });
  EXPECT_NO_THROW({ pool2d(f16, 1, 1, 1, 1, 1, 1); });
  EXPECT_NO_THROW({ embedding(f16, f32); }); // lookup is of a different type
  // Ternary operators
  auto f32_2 = Variable(af::randu({2, 2}, af::dtype::f32), true);
  auto f16_2 = Variable(af::randu({2, 2}, af::dtype::f16), true);
  EXPECT_THROW({ linear(f16, f32, f16_2); }, std::invalid_argument); // linear
  EXPECT_THROW({ linear(f16, f32, f32_2); }, std::invalid_argument); // linear
  auto w = Variable(af::randu(1, af::dtype::f32), true);
  auto b = Variable(af::randu(1, af::dtype::f32), true);
  EXPECT_THROW(
      { batchnorm(f16, f32, f32_2, w, b, {1}, true, 0.01, 0.01); },
      std::invalid_argument);
  EXPECT_THROW(
      { batchnorm(f16, f32, f16_2, w, b, {1}, true, 0.01, 0.01); },
      std::invalid_argument);
  EXPECT_THROW(
      { conv2d(f16, f32, f16_2, 1, 1, 0, 0, 1, 1); }, std::invalid_argument);
  // Quaternary
  auto f16_3 = Variable(af::randu(2, 2, 3, af::dtype::f16), false);
  auto f16_4 = Variable(af::randu(50, af::dtype::f16), false);
  EXPECT_THROW(
      {
        rnn(f16_3,
            Variable(af::array(af::dtype::f32), false),
            Variable(af::array(af::dtype::f32), false),
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

TEST(AutogradTest, CastingAs) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  int refCount = 0;
  // Casting in place should reset refcounts
  auto in = af::randu({5, 5});
  auto var = Variable(in, true);
  af_get_data_ref_count(&refCount, var.array().get());
  ASSERT_EQ(refCount, 2);
  auto varF16 = var.as(af::dtype::f16);
  af_get_data_ref_count(&refCount, varF16.array().get());
  ASSERT_EQ(refCount, 1);
  varF16.backward();
  af_get_data_ref_count(&refCount, varF16.grad().array().get());
  ASSERT_EQ(refCount, 1);
  ASSERT_EQ(varF16.grad().type(), af::dtype::f16);
  ASSERT_EQ(var.grad().type(), af::dtype::f32);

  ASSERT_NE(varF16.type(), in.type());
  ASSERT_TRUE(allClose(varF16.array(), in.as(af::dtype::f16)));
}

TEST(AutogradTest, CastingAsInPlace) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  int refCount = 0;
  auto a = Variable(af::randu({4, 4}), true);
  af_get_data_ref_count(&refCount, a.array().get());
  ASSERT_EQ(refCount, 1);
  a = a.as(af::dtype::f16);
  af_get_data_ref_count(&refCount, a.array().get());
  ASSERT_EQ(refCount, 1);

  ASSERT_EQ(a.type(), af::dtype::f16);
  auto b = Variable(af::randu({4, 4}, af::dtype::f16), true);
  auto c = b + a;
  c.backward();
  ASSERT_EQ(a.grad().type(), af::dtype::f16);

  a = a.as(af::dtype::f32);
  af_get_data_ref_count(&refCount, a.array().get());
  ASSERT_EQ(refCount, 1);
  ASSERT_FALSE(a.isGradAvailable());
}

TEST(AutogradTest, CastingAsDifferentGradTypes) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto f32 = Variable(af::randu({5, 5}), true);
  auto f16 = Variable(af::randu({5, 5}, af::dtype::f16), true);
  // Computing gradients with mixed types fails when the op is applied
  ASSERT_THROW({ f32 + f16; }, std::invalid_argument);
}

TEST(AutogradTest, CastingAsGrad) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // compare to f32 case
  auto x = Variable(af::constant(2.0, 5), true);
  auto y = Variable(af::constant(3.0, 5), true);
  auto z = x * x + x * y + y * y;
  auto dz = Variable(af::constant(1.0, 5), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();

  // f16 -- cast gradients in both directions
  auto x32 = Variable(af::constant(2.0, 5), true);
  auto y32 = Variable(af::constant(3.0, 5), true);
  auto xf16 = x32.as(af::dtype::f16);
  auto yf16 = y32.as(af::dtype::f16);
  auto zf16 = xf16 * xf16 + xf16 * yf16 + yf16 * yf16;
  auto zf32 = zf16.as(af::dtype::f32);
  zf32.backward(dz);

  ASSERT_EQ(xf16.grad().type(), af::dtype::f16);
  ASSERT_EQ(yf16.grad().type(), af::dtype::f16);
  ASSERT_EQ(zf16.grad().type(), af::dtype::f16);
  ASSERT_EQ(x32.grad().type(), af::dtype::f32);
  ASSERT_EQ(y32.grad().type(), af::dtype::f32);
  ASSERT_TRUE(allClose(dx.array(), xf16.grad().array().as(af::dtype::f32)));
  ASSERT_TRUE(allClose(dy.array(), y32.grad().array().as(af::dtype::f32)));
  ASSERT_TRUE(allClose(dx.array(), x32.grad().array()));
  ASSERT_TRUE(allClose(dy.array(), y32.grad().array()));
}

TEST(AutogradTest, NoCalcGrad) {
  auto x = Variable(af::randu(5), false);
  auto y = Variable(af::randu(5), true);
  auto z = x * x + x * y + y * y;
  auto dz = Variable(af::constant(1.0, 5), false);
  z.backward(dz);
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dy.array(), 2 * y.array() + x.array()));
  ASSERT_THROW(x.grad(), std::logic_error);
}

TEST(AutogradTest, MultiplySub) {
  auto x = Variable(af::randu(5), true);
  auto y = Variable(af::randu(5), true);
  auto z = x * x - x * y;
  auto dz = Variable(af::constant(1.0, 5), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dx.array(), (2 * x.array() - y.array())));
  ASSERT_TRUE(allClose(dy.array(), (-x.array())));
}

TEST(AutogradTest, DivideAdd) {
  auto x = Variable(af::randu(5, af::dtype::f64), true);
  auto y = Variable(af::randu(5, af::dtype::f64), true);
  auto z = x + x / y + y;
  auto dz = Variable(af::constant(1.0, 5, af::dtype::f64), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();
  ASSERT_EQ(z.type(), af::dtype::f64);
  ASSERT_TRUE(allClose(dx.array(), (1.0 + 1.0 / y.array())));
  ASSERT_TRUE(
      allClose(dy.array(), (1.0 - x.array() / (y.array() * y.array()))));
}

TEST(AutogradTest, MultiplyAddScalar) {
  auto x = Variable(af::randu(5), true);
  auto y = Variable(af::randu(5), true);
  auto z = 2 * x + x * y + y;
  auto dz = Variable(af::constant(1.0, 5), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dx.array(), (2.0 + y.array())));
  ASSERT_TRUE(allClose(dy.array(), (1.0 + x.array())));
}

TEST(AutogradTest, Exp) {
  auto x = Variable(af::randu(5), true);
  auto y = exp(x);
  auto dy = Variable(af::constant(1.0, 5), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.array(), (af::exp(x.array()))));
}

TEST(AutogradTest, Pow) {
  {
    auto x = Variable(af::randu(5), true);
    auto y = pow(x, 2);
    auto dy = Variable(af::constant(2.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    ASSERT_TRUE(allClose(dx.array(), (2 * 2 * x.array())));
  }
  {
    auto x = Variable(af::randu(5), true);
    auto y = pow(x, 3);
    auto dy = Variable(af::constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    ASSERT_TRUE(allClose(dx.array(), (3 * af::pow(x.array(), 2))));
  }
}

TEST(AutogradTest, Sigmoid) {
  auto x = Variable(af::randu(5), true);
  auto y = sigmoid(x);
  auto dy = Variable(af::constant(1.0, 5), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.array(), (y.array() * (1 - y.array()))));
  ASSERT_TRUE(allClose(
      dx.array(), (af::sigmoid(x.array()) * (1 - af::sigmoid(x.array())))));
}

TEST(AutogradTest, Tanh) {
  auto x = Variable(af::randu(5), true);
  auto y = tanh(x);
  auto dy = Variable(af::constant(1.0, 5), false);
  y.backward(dy);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.array(), (1 - y.array() * y.array())));
  ASSERT_TRUE(allClose(
      dx.array(), (1 + af::tanh(x.array())) * (1 - af::tanh(x.array()))));
}

TEST(AutogradTest, Concatenate) {
  auto x1 = Variable(af::randn(2, 3, 1, 2, f64), true);
  auto x2 = Variable(af::randn(2, 3, 3, 2, f64), true);
  auto x3 = Variable(af::randn(2, 3, 1, 2, f64), true);
  auto x4 = Variable(af::randn(2, 3, 7, 2, f64), true);
  std::vector<Variable> inputs = {x1, x2, x3, x4};
  auto output = concatenate(inputs, 2);

  ASSERT_EQ(output.dims(), af::dim4(2, 3, 12, 2));

  auto func_concatenate_t1 = [x2, x3, x4](Variable& in) {
    return concatenate({in, x2, x3, x4}, 2);
  };
  ASSERT_TRUE(jacobianTestImpl(func_concatenate_t1, x1));

  auto func_concatenate_t2 = [x1, x2, x4](Variable& in) {
    return concatenate({x1, x2, in, x4}, 2);
  };
  ASSERT_TRUE(jacobianTestImpl(func_concatenate_t2, x3));
}

TEST(AutogradTest, Split) {
  // check output
  auto x = Variable(af::range(af::dim4(7, 2)), true);
  auto yVec = split(x, 1, 0);
  ASSERT_EQ(yVec.size(), 7);
  ASSERT_EQ(yVec[0].dims(), af::dim4(1, 2));
  ASSERT_EQ(yVec[2].dims(), af::dim4(1, 2));
  ASSERT_TRUE(af::allTrue<bool>(yVec[6].array() == 6));

  auto a = Variable(af::range(af::dim4(5, 3), 1), true);
  auto bVec = split(a, {2, 1}, 1);
  ASSERT_EQ(bVec.size(), 2);
  ASSERT_EQ(bVec[0].dims(), af::dim4(5, 2));
  ASSERT_EQ(bVec[1].dims(), af::dim4(5, 1));
  ASSERT_TRUE(
      af::allTrue<bool>(bVec[0].array() == af::range(af::dim4(5, 2), 1)));
  ASSERT_TRUE(af::allTrue<bool>(bVec[1].array() == 2));

  // check exception handling
  ASSERT_THROW(split(a, {2, 2}, 0), std::invalid_argument);

  // check gradient
  auto gradFunc = [](Variable& in) { return split(in, 2, 1)[0]; };
  auto input = Variable(af::randu(2, 3, af::dtype::f64), true);
  ASSERT_TRUE(jacobianTestImpl(gradFunc, input));
}

TEST(AutogradTest, TileAs) {
  auto x = Variable(af::randu(5), true);
  auto y = Variable(af::randu(5, 2), true);
  auto z = y * tileAs(x, y);
  auto dz = Variable(af::constant(1.0, 5, 2), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dy.array(), af::tile(x.array(), 1, 2)));
  ASSERT_TRUE(allClose(dx.array(), af::sum(y.array(), 1)));
}

TEST_F(AutogradTestF16, TileAsF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto x = Variable(af::randu(5, af::dtype::f16), true);
  auto y = Variable(af::randu(5, 2, af::dtype::f16), true);
  auto z = y * tileAs(x, y);
  ASSERT_EQ(x.type(), z.type());
  auto dz = Variable(af::constant(1.0, 5, 2, af::dtype::f16), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(
      allClose(dy.array(), af::tile(x.array(), 1, 2).as(dx.type()), 1e-2));
  ASSERT_TRUE(allClose(dx.array(), af::sum(y.array(), 1).as(dx.type()), 1e-2));
}

TEST(AutogradTest, TileAs2) {
  auto x = Variable(af::randu(10), true);
  auto z = tileAs(x, af::dim4(10, 3));
  auto dz = Variable(af::constant(1.0, 10, 3), false);
  z.backward(dz);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.array(), af::constant(3, x.dims())));
}

TEST(AutogradTest, Tile) {
  auto x = Variable(af::randu(6), true);
  auto y = Variable(af::randu(6, 3), true);
  auto z = y * tile(x, af::dim4(1, 3));
  auto dz = Variable(af::constant(1.0, 6, 3), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dy.array(), af::tile(x.array(), 1, 3)));
  ASSERT_TRUE(allClose(dx.array(), af::sum(y.array(), 1)));

  // Jacobian
  auto input = Variable(af::randu(10, 1, 5), true);
  auto func_tile = [](Variable& in) { return tile(in, {1, 2}); };
  ASSERT_TRUE(jacobianTestImpl(func_tile, input, 1E-4, 1E-3));
}

TEST(AutogradTest, Clamp) {
  auto input = Variable(af::randu(5, 6, 7, 4, af::dtype::f64) * 3, true);
  double lo = -1.0, hi = 1.0;
  float perturb = 1E-5;
  // Need to do this as gradient is not continuous when input = lo / hi.
  auto& inarr = input.array();
  af::replace(inarr, af::abs(inarr - lo) > perturb, lo + 10 * perturb);
  af::replace(inarr, af::abs(inarr - hi) > perturb, hi + 10 * perturb);

  auto func_col = [lo, hi](Variable& in) { return clamp(in, lo, hi); };

  ASSERT_TRUE(jacobianTestImpl(func_col, input, 1E-10, perturb));
}

TEST(AutogradTest, SumAs) {
  auto x = Variable(af::randu(5), true);
  auto y = Variable(af::randu(5, 2), true);
  auto z = x * sumAs(y, x);
  auto dz = Variable(af::constant(1.0, 5), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dy.array(), af::tile(x.array(), 1, 2)));
  ASSERT_TRUE(allClose(dx.array(), af::sum(y.array(), 1)));
}

TEST(AutogradTest, SumAs2) {
  auto y = Variable(af::randu(5, 2), true);
  auto z = sumAs(y, af::dim4(5));
  auto dz = Variable(af::constant(1.0, 5), false);
  z.backward(dz);
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dy.array(), af::constant(1.0, 5, 2)));
}

TEST(AutogradTest, Sum) {
  auto x = Variable(af::randu(6), true);
  auto y = Variable(af::randu(6, 3), true);
  auto z = x * sum(y, {1});
  auto dz = Variable(af::constant(1.0, 6), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dy.array(), af::tile(x.array(), 1, 3)));
  ASSERT_TRUE(allClose(dx.array(), af::sum(y.array(), 1)));
}

TEST(AutogradTest, Log1p) {
  auto x = Variable(af::randu(5), true);
  auto y = log1p(x);

  auto xCopy = Variable(x.array(), true);
  auto yExp = log(1 + xCopy);

  y.backward();
  yExp.backward();

  ASSERT_TRUE(allClose(y.array(), yExp.array()));
  ASSERT_TRUE(allClose(y.grad().array(), yExp.grad().array()));
  ASSERT_TRUE(allClose(x.grad().array(), xCopy.grad().array()));
}

TEST(AutogradTest, Sqrt) {
  auto x = Variable(af::randu(5, 3, af::dtype::f64), true);
  auto func_sqrt = [](Variable& in) { return fl::sqrt(in); };
  ASSERT_TRUE(jacobianTestImpl(func_sqrt, x, 1E-3));
}

TEST(AutogradTest, Mean) {
  auto x = Variable(af::randu(5), true);
  auto y = Variable(af::randu(5, 3, 2), true);
  auto z = x * mean(y, {1, 2});
  auto dz = Variable(af::constant(1.0, 5), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dy.array(), af::tile(x.array(), 1, 3, 2) / 6));
  ASSERT_TRUE(allClose(dx.array(), af::mean(af::mean(y.array(), 1), 2)));

  x = Variable(af::randu(5, 3, 2, af::dtype::f64), true);
  auto func_mean = [](Variable& in) { return mean(in, {1, 2}); };
  ASSERT_TRUE(jacobianTestImpl(func_mean, x, 1E-4));
}

TEST(AutogradTest, Variance) {
  auto x = Variable(af::randu(5, 6, 7, 8, af::dtype::f64), true);
  std::vector<bool> biased = {true, false};
  for (auto b : biased) {
    // Behavior of the bias parameter in af::var was changed in
    // https://git.io/Jv5gF and is different in ArrayFire v3.7. If isbiased is
    // true, sample variance rather than population variance is used. The
    // flashlight API implements the opposite behavior to be consistent with
    // other libraries.
    bool afVarBiasArg = !b;

    auto expected_var = af::var(x.array(), afVarBiasArg, 1);
    auto calculated_var = var(x, {1}, b);
    ASSERT_TRUE(allClose(calculated_var.array(), expected_var));

    auto func_var = [b](Variable& in) { return var(in, {1, 2}, b); };
    ASSERT_TRUE(jacobianTestImpl(func_var, x, 1E-5, 1E-5));
  }
}

TEST(AutogradTest, Norm) {
  auto x = Variable(af::randu(5, 3, af::dtype::f64), true);
  auto funcNorm2 = [](Variable& in) { return norm(in, {1}); };
  ASSERT_TRUE(jacobianTestImpl(funcNorm2, x, 1E-4));
  auto funcNorm1 = [](Variable& in) { return norm(in, {1}, 1); };
  ASSERT_TRUE(jacobianTestImpl(funcNorm1, x, 1E-4));
  auto funcNorm3 = [](Variable& in) { return norm(in, {1}, 3); };
  ASSERT_TRUE(jacobianTestImpl(funcNorm3, x, 1E-4));
}

TEST(AutogradTest, Normalize) {
  auto x = Variable(af::randu(5, 3, af::dtype::f64), true);
  auto funcNormalize2 = [](Variable& in) { return normalize(in, {1}); };
  ASSERT_TRUE(jacobianTestImpl(funcNormalize2, x));
  auto ys = funcNormalize2(x);
  ASSERT_TRUE(allClose(
      af::sum(ys.array() * ys.array(), 1), af::constant(1, 5, af::dtype::f64)));
  auto yb = normalize(x, {1}, 2, 1);
  ASSERT_TRUE(
      af::allTrue<bool>(af::sqrt(af::sum(yb.array() * yb.array(), 1)) <= 1));
}

TEST(AutogradTest, Indexing) {
  auto x = Variable(af::randu(5, 6, 7, 4, af::dtype::f64), true);

  auto func_col = [](Variable& input) { return input.col(4); };
  ASSERT_TRUE(jacobianTestImpl(func_col, x));

  auto func_row = [](Variable& input) { return input.row(4); };
  ASSERT_TRUE(jacobianTestImpl(func_row, x));

  auto func_slice = [](Variable& input) { return input.slice(4); };
  ASSERT_TRUE(jacobianTestImpl(func_slice, x));

  auto func_cols = [](Variable& input) { return input.cols(2, 4); };
  ASSERT_TRUE(jacobianTestImpl(func_cols, x));

  auto func_rows = [](Variable& input) { return input.rows(2, 4); };
  ASSERT_TRUE(jacobianTestImpl(func_rows, x));

  auto func_slices = [](Variable& input) { return input.slices(2, 4); };
  ASSERT_TRUE(jacobianTestImpl(func_slices, x));
}

TEST(AutogradTest, Convolve) {
  auto in = Variable(af::randu(10, 9, 8, 7, af::dtype::f32), true);
  auto wt = Variable(af::randu(4, 3, 8, 6, af::dtype::f32), true);
  auto bs = Variable(af::randu(1, 1, 6, 1, af::dtype::f32), true);
  int px = 2, py = 1;
  int sx = 1, sy = 1;
  int dx = 1, dy = 1;
  auto benchmarks = std::make_shared<detail::ConvBenchmarks>();
  auto func_conv_in = [&](Variable& input) {
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
  ASSERT_TRUE(jacobianTestImpl(func_conv_in, in, 0.06));
  auto func_conv_wt = [&](Variable& weight) {
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
  ASSERT_TRUE(jacobianTestImpl(func_conv_wt, wt, 0.06));
  auto func_conv_bs = [&](Variable& bias) {
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
  ASSERT_TRUE(jacobianTestImpl(func_conv_bs, bs, 0.03));
}

TEST_F(AutogradTestF16, ConvolveF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  const float scaleFactor = 10.0; // scale the input to prevent grad underflow
  auto in = Variable(af::randu(3, 1, 2, 1, af::dtype::f16) * scaleFactor, true);
  auto wt = Variable(af::randu(2, 1, 2, 1, af::dtype::f16), true);
  auto bs = Variable(af::randu(1, 1, 1, 1, af::dtype::f16), true);
  int px = 1, py = 1;
  int sx = 1, sy = 1;
  int dx = 1, dy = 1;
  auto benchmarks = std::make_shared<detail::ConvBenchmarks>();
  auto func_conv_in = [&](Variable& input) {
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
  ASSERT_TRUE(jacobianTestImpl(func_conv_in, in, 5e-1, 0.1));
  auto func_conv_wt = [&](Variable& weight) {
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
  ASSERT_TRUE(jacobianTestImpl(func_conv_wt, wt, 5e-2, 0.1));
  auto func_conv_bs = [&](Variable& bias) {
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
  ASSERT_TRUE(jacobianTestImpl(func_conv_bs, bs, 3e-2, 0.1));
}

TEST(AutogradTest, ConvolveFilterGroups) {
  int channel = 8;
  int groups = 2;
  // w x h x c x b
  auto in = Variable(af::randu(10, 9, channel, 7, af::dtype::f32), true);
  // w x h x in x out
  auto wt =
      Variable(af::randu(4, 3, channel / groups, 6, af::dtype::f32), true);
  auto bs = Variable(af::randu(1, 1, 6, 1, af::dtype::f32), true);

  int px = 2, py = 1;
  int sx = 1, sy = 1;
  int dx = 1, dy = 1;
  auto func_conv_in = [&](Variable& input) {
    return conv2d(input, wt, bs, sx, sy, px, py, dx, dy, groups);
  };
  ASSERT_TRUE(jacobianTestImpl(func_conv_in, in, 0.06));
  auto func_conv_wt = [&](Variable& weight) {
    return conv2d(in, weight, bs, sx, sy, px, py, dx, dy, groups);
  };
  ASSERT_TRUE(jacobianTestImpl(func_conv_wt, wt, 0.05));
  auto func_conv_bs = [&](Variable& bias) {
    return conv2d(in, wt, bias, sx, sy, px, py, dx, dy, groups);
  };
  ASSERT_TRUE(jacobianTestImpl(func_conv_bs, bs, 0.02));
}

TEST(AutogradTest, ConvolveDilation) {
  auto in = Variable(af::randu(10, 9, 8, 7, af::dtype::f32), true);
  auto wt = Variable(af::randu(4, 3, 8, 6, af::dtype::f32), true);
  auto bs = Variable(af::randu(1, 1, 6, 1, af::dtype::f32), true);
  int px = 2, py = 1;
  int sx = 1, sy = 1;
  int dx = 2, dy = 1;
  auto func_conv_in = [&](Variable& input) {
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
  ASSERT_TRUE(jacobianTestImpl(func_conv_in, in, 0.06));
  auto func_conv_wt = [&](Variable& weight) {
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
  ASSERT_TRUE(jacobianTestImpl(func_conv_wt, wt, 0.05));
  auto func_conv_bs = [&](Variable& bias) {
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
  ASSERT_TRUE(jacobianTestImpl(func_conv_bs, bs, 0.02));
}

TEST(AutogradTest, Padding) {
  auto in = Variable(af::randu(3, 3, af::dtype::f32), true);
  auto func_pad = [&](Variable& input) {
    return padding(input, {{1, 2}, {0, 1}}, -1);
  };
  ASSERT_TRUE(jacobianTestImpl(func_pad, in, 1E-3));
}

TEST(AutogradTest, Pooling) {
  auto in = Variable(af::randu(3, 3, 1, 1, af::dtype::f32), true);
  auto func_pool = [&](Variable& input) { return pool2d(input, 2, 2, 1, 1); };
  ASSERT_TRUE(jacobianTestImpl(func_pool, in, 1E-3));
}

TEST_F(AutogradTestF16, PoolingF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  const float inputScale = 2.0; // scale the input to prevent grad underflow
  auto in = Variable(inputScale * af::randu(3, 3, 1, 1, af::dtype::f16), true);
  auto func_pool = [&](Variable& input) { return pool2d(input, 2, 2, 1, 1); };
  ASSERT_TRUE(jacobianTestImpl(func_pool, in, 1e1, 1e-1)); // TODO: investigate
}

TEST(AutogradTest, Softmax) {
  auto in = Variable(af::randu(3, 5, 1, af::dtype::f64), true);
  auto func_sm = [&](Variable& input) { return softmax(input, 0); };

  ASSERT_TRUE(jacobianTestImpl(func_sm, in, 1E-5));
}

TEST_F(AutogradTestF16, SoftmaxF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto in = Variable(af::randu(3, 5, 1, af::dtype::f16), true);
  auto func_sm = [&](Variable& input) { return softmax(input, 0); };

  ASSERT_TRUE(jacobianTestImpl(func_sm, in, 1E-2, 1e-1));
}

TEST(AutogradTest, LogSoftmax) {
  auto in = Variable(af::randu(3, 5, 1, af::dtype::f64), true);
  auto func_lsm = [&](Variable& input) { return logSoftmax(input, 0); };

  ASSERT_TRUE(jacobianTestImpl(func_lsm, in, 1E-5));
}

TEST_F(AutogradTestF16, LogSoftmaxF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto in = Variable(af::randu(3, 5, 1, af::dtype::f16), true);
  auto func_lsm = [&](Variable& input) { return logSoftmax(input, 0); };

  ASSERT_TRUE(jacobianTestImpl(func_lsm, in, 1E-2, 1e-1));
}

TEST(AutogradTest, BinaryCrossEntropy) {
  auto x = Variable(af::randu(10), true);
  auto y = Variable(af::randu(10), true);
  auto loss = binaryCrossEntropy(x, y);

  // bce loss should be positive
  ASSERT_TRUE(af::allTrue<bool>(loss.array() > 0));
}

TEST(AutogradTest, CrossEntropy) {
  auto x = Variable(af::randu(7, 10, 4, af::dtype::f64), true);
  auto y = Variable((af::randu(10, 4, af::dtype::u32) % 7).as(s32), false);
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
  auto ignoreCount = af::sum<int>(y.array() == ignoreIdx);
  ASSERT_NEAR(
      (lossSumIgnore / lossMeanIgnore).scalar<double>(),
      40 - ignoreCount,
      1e-5);
}

TEST(AutogradTest, Reorder) {
  auto in = Variable(af::randu(3, 1, 4, 1, af::dtype::f32) * 2, true);
  auto func_reorder = [&](Variable& input) {
    return reorder(input, 2, 0, 3, 1);
  };
  ASSERT_TRUE(jacobianTestImpl(func_reorder, in, 1E-3));
}

TEST(AutogradTest, Glu) {
  auto in = Variable(af::randu(3, 4, 5, af::dtype::f64), true);
  auto func_glu = [&](Variable& input) { return gatedlinearunit(input, 1); };
  ASSERT_TRUE(jacobianTestImpl(func_glu, in, 1E-5));
}

TEST(AutogradTest, Linear) {
  std::vector<int> batchsizes = {1, 5};
  for (auto b : batchsizes) {
    auto in = Variable(af::randu(3, 4, b, af::dtype::f64) * 2 - 1, true);
    auto wt = Variable(af::randu(6, 3, af::dtype::f64) * 2 - 1, true);
    auto bs = Variable(af::randu(6, af::dtype::f64) * 2 - 1, true);
    auto func_lin_in = [&](Variable& input) { return linear(input, wt, bs); };
    ASSERT_TRUE(jacobianTestImpl(func_lin_in, in, 1E-8));
    auto func_lin_wt = [&](Variable& weight) { return linear(in, weight, bs); };
    ASSERT_TRUE(jacobianTestImpl(func_lin_wt, wt, 1E-8));
    auto func_lin_bs = [&](Variable& bias) { return linear(in, wt, bias); };
    ASSERT_TRUE(jacobianTestImpl(func_lin_bs, bs, 1E-8));
  }
}

TEST_F(AutogradTestF16, LinearF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  std::vector<int> batchsizes = {1, 5};
  const float scale = 4.0; // scale prevent grad underflow
  for (auto b : batchsizes) {
    auto in = Variable(af::randu(2, 2, b, af::dtype::f16) * scale, true);
    auto wt = Variable(af::randu(2, 2, af::dtype::f16) * scale, true);
    auto bs = Variable(af::randu(2, af::dtype::f16) * scale, true);
    auto func_lin_in = [&](Variable& input) { return linear(input, wt, bs); };
    ASSERT_TRUE(jacobianTestImpl(func_lin_in, in, 5E-2, 5E-1));
    auto func_lin_wt = [&](Variable& weight) { return linear(in, weight, bs); };
    ASSERT_TRUE(jacobianTestImpl(func_lin_wt, wt, 5E-2, 5E-1));
    auto func_lin_bs = [&](Variable& bias) { return linear(in, wt, bias); };
    ASSERT_TRUE(jacobianTestImpl(func_lin_bs, bs, 5E-2, 5E-1));
  }
}

TEST(AutogradTest, WeightNormLinear) {
  auto v = Variable(af::randu(3, 2), true);
  auto norm_dim = {1};
  auto g = Variable(norm(v, norm_dim).array(), true);
  auto in = Variable(af::randu(2, 3, 1, 1, af::dtype::f32), true);

  auto func_weightNorm_in = [&](Variable& input) {
    auto w = v * tileAs(g / norm(v, norm_dim), v);
    return matmul(w, input);
  };
  ASSERT_TRUE(jacobianTestImpl(func_weightNorm_in, in, 1E-3));

  auto func_weightNorm_v = [&](Variable& input) {
    auto w = input * tileAs(g / norm(input, norm_dim), input);
    return matmul(w, in);
  };
  ASSERT_TRUE(jacobianTestImpl(func_weightNorm_v, v, 1E-2));

  auto func_weightNorm_g = [&](Variable& input) {
    auto w = v * tileAs(input / norm(v, norm_dim), v);
    return matmul(w, in);
  };
  ASSERT_TRUE(jacobianTestImpl(func_weightNorm_g, g, 1E-3));
}

TEST(AutogradTest, WeightNormConv) {
  auto v = Variable(af::randu(3, 3, 3, 8), true);
  auto norm_dim = {0, 1, 2};
  auto g = Variable(norm(v, norm_dim).array(), true);
  auto in = Variable(af::randu(7, 7, 3, 8) * 2 - 2, true);

  auto func_weightNorm_in = [&](Variable& input) {
    auto w = v * tileAs(g / norm(v, norm_dim), v);
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
  ASSERT_TRUE(jacobianTestImpl(func_weightNorm_in, in, 3E-1));

  auto func_weightNorm_v = [&](Variable& input) {
    auto w = input * tileAs(g / norm(input, norm_dim), input);
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
  ASSERT_TRUE(jacobianTestImpl(func_weightNorm_v, v, 2E-1));

  auto func_weightNorm_g = [&](Variable& input) {
    auto w = v * tileAs(input / norm(v, norm_dim), v);
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
  ASSERT_TRUE(jacobianTestImpl(func_weightNorm_g, g, 2E-1));
}

constexpr af::dtype kDefaultRnnPrecision =
    FL_BACKEND_MIOPEN ? af::dtype::f32 : af::dtype::f64;

void testRnnImpl(RnnMode mode, af::dtype precision = kDefaultRnnPrecision) {
  int numLayers = 2;
  int hiddenSize = 2;
  int inputSize = 2;
  int batchSize = 2;
  int seqLength = 3;
  bool bidirectional = true;
  float expectedPrecision = 0;
  float perturbation = 0;
  switch (precision) {
    case af::dtype::f16:
      expectedPrecision = 5E-2;
      perturbation = 1E-1;
      break;
    case af::dtype::f32:
      expectedPrecision = 1E-3;
      perturbation = 1E-2;
      break;
    case af::dtype::f64:
      expectedPrecision = 1E-5;
      perturbation = 1E-4;
      break;
  }

  auto in =
      Variable(af::randu(inputSize, batchSize, seqLength, precision), true);
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

  auto w = Variable(af::randu(nParams, precision), true);

  auto funcRnnIn = [&](Variable& input) -> Variable {
    return std::get<0>(
        rnn(input,
            Variable().as(precision),
            Variable().as(precision),
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
            Variable().as(precision),
            Variable().as(precision),
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
      af::randu(
          inputSize,
          batchSize,
          numLayers * (1 + bidirectional),
          af::dtype::f64),
      true);
  auto funcRnnHx = [&](Variable& hiddenState) -> Variable {
    return std::get<0>(
        rnn(in,
            hiddenState.as(precision),
            Variable().as(precision),
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
            Variable().as(precision),
            Variable().as(precision),
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
        af::randu(
            inputSize,
            batchSize,
            numLayers * (1 + bidirectional),
            af::dtype::f64),
        true);
    auto funcRnnCx = [&](Variable& cellState) -> Variable {
      return std::get<0>(
          rnn(in,
              Variable().as(precision),
              cellState.as(precision),
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
              Variable().as(precision),
              Variable().as(precision),
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

  testRnnImpl(RnnMode::TANH, af::dtype::f16);
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

  testRnnImpl(RnnMode::LSTM, af::dtype::f16);
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

  testRnnImpl(RnnMode::GRU, af::dtype::f16);
}

TEST(AutogradTest, Embedding) {
  int n_words = 10;
  auto input = Variable((af::randu(4, 2) * n_words).as(s32), false);
  auto weights = Variable(af::randn(4, n_words, f64), true);
  auto func_embed = [&](Variable& w) { return embedding(input, w); };
  ASSERT_TRUE(jacobianTestImpl(func_embed, weights, 1E-5));
}

TEST(AutogradTest, BatchNormEvalModeOutputSingleAxis) {
  int feat_dims = 3;
  std::vector<int> featAxes = {2};
  // input order: HWCN, following the docs
  auto input = Variable(af::randu(13, 13, feat_dims, 16), false);
  auto runningMean = Variable(af::randu(feat_dims, input.type()), false);
  auto runningVar = Variable(af::randu(feat_dims, input.type()), false);
  auto weight = Variable(af::randu(feat_dims, input.type()), false);
  auto bias = Variable(af::randu(feat_dims, input.type()), false);

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
  for (int i = 0; i < feat_dims; ++i) {
    std::array<af::index, 4> sel = {af::span, af::span, i, af::span};
    auto thisInput = input.array()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.array()(i).scalar<float>();
    auto thisVar = runningVar.array()(i).scalar<float>();
    auto thisWeight = weight.array()(i).scalar<float>();
    auto thisBias = bias.array()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / sqrt(thisVar + 1E-5);
    expectedOut = expectedOut * thisWeight + thisBias;
    ASSERT_TRUE(allClose(
        out.array()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1E-5));
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
  for (int i = 0; i < feat_dims; ++i) {
    std::array<af::index, 4> sel = {af::span, af::span, i, af::span};
    auto thisInput = input.array()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.array()(i).scalar<float>();
    auto thisVar = runningVar.array()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / sqrt(thisVar + 1E-5);
    ASSERT_TRUE(allClose(
        out.array()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1E-5));
  }
}

TEST(AutogradTest, BatchNormEvalModeOutputMultipleAxis) {
  // input order: HWCN, following the docs
  std::vector<int> featAxes = {0, 1, 2};
  auto input = Variable(af::randu(13, 13, 4, 16), false);

  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dims(ax);
  }
  auto runningMean = Variable(af::randu(nfeatures, input.type()), false);
  auto runningVar = Variable(af::randu(nfeatures, input.type()), false);
  auto weight = Variable(af::randu(nfeatures, input.type()), false);
  auto bias = Variable(af::randu(nfeatures, input.type()), false);

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
    std::array<af::index, 4> sel = {
        i % 13, (i / 13) % 13, (i / 13) / 13, af::span};
    auto thisInput = input.array()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.array()(i).scalar<float>();
    auto thisVar = runningVar.array()(i).scalar<float>();
    auto thisWeight = weight.array()(i).scalar<float>();
    auto thisBias = bias.array()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / sqrt(thisVar + 1e-5);
    expectedOut = expectedOut * thisWeight + thisBias;
    ASSERT_TRUE(allClose(
        out.array()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1e-5));
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
    std::array<af::index, 4> sel = {
        i % 13, (i / 13) % 13, (i / 13) / 13, af::span};
    auto thisInput = input.array()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = runningMean.array()(i).scalar<float>();
    auto thisVar = runningVar.array()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / sqrt(thisVar + 1e-5);
    ASSERT_TRUE(allClose(
        out.array()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1e-5));
  }
}

TEST(AutogradTest, BatchNormTrainModeOutputSingleAxis) {
  int numFeat = 3;
  std::vector<int> featAxes = {2};
  double epsilon = 1E-5;
  auto input = Variable(af::randu(13, 13, numFeat, 8), true);
  auto weight = Variable(af::randu(numFeat), true);
  auto bias = Variable(af::randu(numFeat), true);
  auto runningMean = Variable(af::randu(numFeat), false);
  auto runningVar = Variable(af::randu(numFeat), false);

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

  auto todim = af::dim4(1, 1, numFeat);
  std::vector<int> nrm_axes = {0, 1, 3};
  auto avg = moddims(mean(input, nrm_axes), todim);
  auto variance =
      moddims(var(input, nrm_axes, true /* population var */), todim);
  auto expectedOut = (input - tileAs(avg, input)) /
      fl::sqrt(tileAs(variance, input) + epsilon);
  expectedOut = expectedOut * tileAs(moddims(weight, todim), input) +
      tileAs(moddims(bias, todim), input);
  ASSERT_TRUE(allClose(out.array(), expectedOut.array(), 1e-5));
}

TEST(AutogradTest, BatchNormTrainModeOutputMultipleAxis) {
  std::vector<int> featAxes = {0, 1, 2};
  auto input = Variable(af::randu(13, 13, 4, 8), true);

  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dims(ax);
  }
  auto weight = Variable(af::randu(nfeatures), true);
  auto bias = Variable(af::randu(nfeatures), true);
  auto runningMean = Variable(af::randu(nfeatures), false);
  auto runningVar = Variable(af::randu(nfeatures), false);

  auto out = batchnorm(
      input, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5);

  auto todim = af::dim4(nfeatures);
  std::vector<int> nrm_axes = {3};
  auto avg = moddims(mean(input, nrm_axes), todim);
  auto variance = moddims(var(input, nrm_axes, true), todim);

  for (int i = 0; i < nfeatures; ++i) {
    std::array<af::index, 4> sel = {
        i % 13, (i / 13) % 13, (i / 13) / 13, af::span};
    auto thisInput = input.array()(sel[0], sel[1], sel[2], sel[3]);
    auto thisMean = avg(i).scalar<float>();
    auto thisVar = variance(i).scalar<float>();
    auto thisWeight = weight.array()(i).scalar<float>();
    auto thisBias = bias.array()(i).scalar<float>();

    auto expectedOut = (thisInput - thisMean) / sqrt(thisVar + 1e-5);
    expectedOut = expectedOut * thisWeight + thisBias;
    ASSERT_TRUE(allClose(
        out.array()(sel[0], sel[1], sel[2], sel[3]), expectedOut, 1e-5));
  }
}

TEST(AutogradTest, BatchNormJacobian) {
  // Jacobian Test with  train_mode = true;

  int numFeat = 3;
  std::vector<int> featAxes = {2};
  auto input = Variable(af::randu(8, 8, numFeat, 16, af::dtype::f32), true);
  auto runningMean = Variable(af::randu(numFeat, af::dtype::f32), false);
  auto runningVar = Variable(af::randu(numFeat, af::dtype::f32), false);
  auto weight = Variable(af::randu(numFeat, af::dtype::f32), true);
  auto bias = Variable(af::randu(numFeat, af::dtype::f32), true);

  auto func_bn_in = [&](Variable& in) {
    return (batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(func_bn_in, input, 1e-2, 1e-4));

  auto func_bn_wt = [&](Variable& wt) {
    return (batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(func_bn_wt, weight, 1e-2, 1e-4));

  auto func_bn_bs = [&](Variable& bs) {
    return (batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(func_bn_bs, bias, 1e-2, 1e-4));
}

TEST_F(AutogradTestF16, BatchNormJacobianF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // Jacobian Test with  train_mode = true;

  int numFeat = 3;
  std::vector<int> featAxes = {2};
  auto input = Variable(af::randu(8, 8, numFeat, 16, af::dtype::f16), true);
  auto runningMean = Variable(af::randu(numFeat, af::dtype::f32), false);
  auto runningVar = Variable(af::randu(numFeat, af::dtype::f32), false);
  auto weight = Variable(af::randu(numFeat, af::dtype::f32), true);
  auto bias = Variable(af::randu(numFeat, af::dtype::f32), true);

  // Use larger perturbations to ensure gradients don't underflow with fp16

  auto func_bn_in = [&](Variable& in) {
    return (batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(func_bn_in, input, 5e-2, 1e-1));

  auto func_bn_wt = [&](Variable& wt) {
    return (batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(func_bn_wt, weight, 5e-2, 1e-1));

  auto func_bn_bs = [&](Variable& bs) {
    return (batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(func_bn_bs, bias, 5e-2, 1e-1));
}

TEST(AutogradTest, BatchNormJacobianMultipleAxes) {
  // Jacobian Test with  train_mode = true;
  std::vector<int> featAxes = {0, 1, 2};
  auto input = Variable(af::randu(8, 8, 3, 16, af::dtype::f32), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dims(ax);
  }
  auto runningMean = Variable(af::randu(nfeatures, af::dtype::f32), false);
  auto runningVar = Variable(af::randu(nfeatures, af::dtype::f32), false);
  auto weight = Variable(af::randu(nfeatures, af::dtype::f32), true);
  auto bias = Variable(af::randu(nfeatures, af::dtype::f32), true);

  auto func_bn_in = [&](Variable& in) {
    return (batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(func_bn_in, input, 1e-2, 1e-3));

  auto func_bn_wt = [&](Variable& wt) {
    return (batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(func_bn_wt, weight, 1e-2, 1e-3));

  auto func_bn_bs = [&](Variable& bs) {
    return (batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(func_bn_bs, bias, 1e-2, 1e-3));
}

TEST_F(AutogradTestF16, BatchNormJacobianMultipleAxesF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // Jacobian Test with  train_mode = true;
  std::vector<int> featAxes = {0, 1, 2};
  auto input = Variable(af::randu(2, 2, 2, 1, af::dtype::f16), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dims(ax);
  }
  auto runningMean = Variable(af::randu(nfeatures, af::dtype::f32), false);
  auto runningVar = Variable(af::randu(nfeatures, af::dtype::f32), false);
  auto weight = Variable(af::randu(nfeatures, af::dtype::f32), true);
  auto bias = Variable(af::randu(nfeatures, af::dtype::f32), true);

  // Use larger perturbations to ensure gradients don't underflow with fp16

  auto func_bn_in = [&](Variable& in) {
    return (batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(
      jacobianTestImpl(func_bn_in, input, 5e-2, 1e-1)); // TODO: investigate

  auto func_bn_wt = [&](Variable& wt) {
    return (batchnorm(
        input, wt, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(func_bn_wt, weight, 5e-2, 1e-1));

  auto func_bn_bs = [&](Variable& bs) {
    return (batchnorm(
        input, weight, bs, runningMean, runningVar, featAxes, true, 0.0, 1E-5));
  };
  ASSERT_TRUE(jacobianTestImpl(func_bn_bs, bias, 5e-2, 1e-1));
}

TEST(AutogradTest, LayerNormJacobian) {
  std::vector<int> featAxes = {0, 1, 2, 3};
  auto input = Variable(af::randu(7, 7, 3, 10), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dims(ax);
  }
  auto runningMean = Variable(af::randu(nfeatures, af::dtype::f32), false);
  auto runningVar = Variable(af::randu(nfeatures, af::dtype::f32), false);
  auto weight = Variable(af::randu(nfeatures, af::dtype::f32), true);
  auto bias = Variable(af::randu(nfeatures, af::dtype::f32), true);

  auto func_ln_in = [&](Variable& in) {
    return batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5);
  };

  ASSERT_TRUE(jacobianTestImpl(func_ln_in, input, 1e-2, 1e-4));
}

TEST_F(AutogradTestF16, LayerNormJacobianF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  std::vector<int> featAxes = {0, 1, 2, 3};
  const float inputScale = 4.0; // scale the input to prevent grad underflow
  auto input =
      Variable(inputScale * af::randu(2, 2, 2, 4, af::dtype::f16), true);
  auto nfeatures = 1;
  for (auto ax : featAxes) {
    nfeatures *= input.dims(ax);
  }
  auto runningMean = Variable(af::randu(nfeatures, af::dtype::f32), false);
  auto runningVar = Variable(af::randu(nfeatures, af::dtype::f32), false);
  auto weight = Variable(af::randu(nfeatures, af::dtype::f32), true);
  auto bias = Variable(af::randu(nfeatures, af::dtype::f32), true);

  auto func_ln_in = [&](Variable& in) {
    return batchnorm(
        in, weight, bias, runningMean, runningVar, featAxes, true, 0.0, 1E-5);
  };

  ASSERT_TRUE(jacobianTestImpl(func_ln_in, input, 1e-4, 1e-2));
}

TEST(AutogradTest, GetAdvancedIndex) {
  if (af::getActiveBackend() != AF_BACKEND_CUDA) {
    GTEST_SKIP()
        << "Advanced indexing operator unsupported for non-CUDA backends";
  }
  std::vector<af::dtype> validIndexTypes{s32, s64, u32, u64};
  for (const auto& dtype : validIndexTypes) {
    auto x = Variable(af::randu(20, 50, 40, 30, f32), true);
    af::array a(6, dtype);
    a(0) = 0;
    a(1) = 15;
    a(2) = 6;
    a(3) = 1;
    a(4) = 10;
    a(5) = 6;
    af::array b(3, dtype);
    b(0) = 5;
    b(1) = 11;
    b(2) = 19;
    auto x2 = x(a, b, af::span, af::seq(0, 3));
    auto y = sum(x2 * x2, {0, 1, 2, 3});
    auto res = 2 * sum(x2, {0, 1, 2, 3}).array();
    y.backward();
    auto grad = sum(x.grad(), {0, 1, 2, 3}).array();
    ASSERT_TRUE(allClose(grad, res, 1e-3));
  }
}

TEST(AutogradTest, GetAdvancedIndexF16) {
  if (af::getActiveBackend() != AF_BACKEND_CUDA) {
    GTEST_SKIP()
        << "Advanced indexing operator unsupported for non-CUDA backends";
  }
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }
  std::vector<af::dtype> validIndexTypes{s32, s64, u32, u64};
  for (const auto& dtype : validIndexTypes) {
    auto x = Variable(af::randu(20, 50, 40, 30, f16), true);
    af::array a(6, dtype);
    a(0) = 0;
    a(1) = 15;
    a(2) = 6;
    a(3) = 1;
    a(4) = 10;
    a(5) = 6;
    af::array b(3, dtype);
    b(0) = 5;
    b(1) = 11;
    b(2) = 19;
    auto x2 = x(a, b, af::span, af::seq(0, 3));
    ASSERT_EQ(x2.type(), af::dtype::f16);
    auto y = sum(x2 * x2, {0, 1, 2, 3});
    auto res = 2 * sum(x2, {0, 1, 2, 3}).array();
    y.backward();
    ASSERT_EQ(x.grad().type(), af::dtype::f16);
    auto grad = sum(x.grad(), {0, 1, 2, 3}).array();
    ASSERT_TRUE(allClose(grad, res, 1e-3));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
