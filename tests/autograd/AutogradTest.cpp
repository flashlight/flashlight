/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
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
#include "flashlight/autograd/autograd.h"
#include "flashlight/common/common.h"

using namespace fl;

namespace {

using JacobianFunc = std::function<Variable(Variable&)>;
bool jacobianTestImpl(
    const JacobianFunc& func,
    Variable& input,
    float precision = 1E-5,
    float perturbation = 1E-4) {
  auto fwd_jacobian =
      af::array(func(input).elements(), input.elements(), af::dtype::f32);

  for (int i = 0; i < input.elements(); ++i) {
    af::array orig = input.array()(i);
    input.array()(i) = orig - perturbation;
    auto outa = func(input).array();

    input.array()(i) = orig + perturbation;
    auto outb = func(input).array();
    input.array()(i) = orig;
    fwd_jacobian(af::span, i) =
        af::moddims((outb - outa), outa.elements()) * 0.5 / perturbation;
  }

  auto bwd_jacobian =
      af::array(func(input).elements(), input.elements(), af::dtype::f32);
  auto dout =
      Variable(af::constant(0, func(input).dims(), input.type()), false);
  for (int i = 0; i < dout.elements(); ++i) {
    dout.array()(i) = 1; // element in 1D view
    input.zeroGrad();
    auto out = func(input);
    out.backward(dout);
    bwd_jacobian(i, af::span) =
        af::moddims(input.grad().array(), input.elements());
    dout.array()(i) = 0;
  }
  return allClose(fwd_jacobian, bwd_jacobian, precision);
}

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
  auto dz = Variable(af::constant(1.0, 5), false);
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
    auto expected_var = var(x.array(), b, 1);
    auto calculated_var = var(x, {1}, b);
    ASSERT_TRUE(allClose(calculated_var.array(), expected_var));

    auto func_var = [b](Variable& in) { return var(in, {1, 2}, b); };
    ASSERT_TRUE(jacobianTestImpl(func_var, x, 1E-5, 1E-5));
  }
}

TEST(AutogradTest, Norm) {
  auto x = Variable(af::randu(5, 3, af::dtype::f64), true);
  auto func_norm = [](Variable& in) { return norm(in, {1}); };
  ASSERT_TRUE(jacobianTestImpl(func_norm, x, 1E-4));
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
  auto func_conv_in = [&](Variable& input) {
    return conv2d(input, wt, bs, sx, sy, px, py);
  };
  ASSERT_TRUE(jacobianTestImpl(func_conv_in, in, 0.06));
  auto func_conv_wt = [&](Variable& weight) {
    return conv2d(in, weight, bs, sx, sy, px, py);
  };
  ASSERT_TRUE(jacobianTestImpl(func_conv_wt, wt, 0.05));
  auto func_conv_bs = [&](Variable& bias) {
    return conv2d(in, wt, bias, sx, sy, px, py);
  };
  ASSERT_TRUE(jacobianTestImpl(func_conv_bs, bs, 0.02));
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

  auto func_conv_in = [&](Variable& input) {
    return conv2d(input, wt, bs, sx, sy, px, py, groups);
  };
  ASSERT_TRUE(jacobianTestImpl(func_conv_in, in, 0.06));
  auto func_conv_wt = [&](Variable& weight) {
    return conv2d(in, weight, bs, sx, sy, px, py, groups);
  };
  ASSERT_TRUE(jacobianTestImpl(func_conv_wt, wt, 0.05));
  auto func_conv_bs = [&](Variable& bias) {
    return conv2d(in, wt, bias, sx, sy, px, py, groups);
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

TEST(AutogradTest, Softmax) {
  auto in = Variable(af::randu(3, 5, 1, af::dtype::f64), true);
  auto func_sm = [&](Variable& input) { return softmax(input, 0); };

  ASSERT_TRUE(jacobianTestImpl(func_sm, in, 1E-5));
}

TEST(AutogradTest, LogSoftmax) {
  auto in = Variable(af::randu(3, 5, 1, af::dtype::f64), true);
  auto func_lsm = [&](Variable& input) { return logSoftmax(input, 0); };

  ASSERT_TRUE(jacobianTestImpl(func_lsm, in, 1E-5));
}

TEST(AutogradTest, BinaryCrossEntropy) {
  auto x = Variable(af::randu(10), true);
  auto y = Variable(af::randu(10), true);
  auto loss = binaryCrossEntropy(x, y);

  // bce loss should be positive
  ASSERT_TRUE(af::allTrue<bool>(loss.array() > 0));
}

TEST(AutogradTest, CrossEntropy) {
  auto in = Variable(af::randu(7, 10, 1, 1, af::dtype::f64), true);
  af::dim4 dims(10, 1, 1, 1);
  auto y = Variable(af::constant(1.0, dims, af::dtype::f64), false);

  auto func_crossent_mean = [&](Variable& input) {
    return categoricalCrossEntropy(input, y);
  };
  ASSERT_TRUE(jacobianTestImpl(func_crossent_mean, in, 1E-5));

  auto func_crossent_noreduce = [&](Variable& input) {
    return categoricalCrossEntropy(input, y, ReduceMode::NONE);
  };
  ASSERT_TRUE(jacobianTestImpl(func_crossent_noreduce, in, 1E-5));
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
    return conv2d(input, w);
  };
  ASSERT_TRUE(jacobianTestImpl(func_weightNorm_in, in, 1E-1));

  auto func_weightNorm_v = [&](Variable& input) {
    auto w = input * tileAs(g / norm(input, norm_dim), input);
    return conv2d(in, w);
  };
  ASSERT_TRUE(jacobianTestImpl(func_weightNorm_v, v, 1E-1));

  auto func_weightNorm_g = [&](Variable& input) {
    auto w = v * tileAs(input / norm(v, norm_dim), v);
    return conv2d(in, w);
  };
  ASSERT_TRUE(jacobianTestImpl(func_weightNorm_g, g, 1E-1));
}

void test_rnn_impl(RnnMode mode) {
  int num_layers = 2;
  int hidden_size = 2;
  int input_size = 2;
  int batch_size = 2;
  int seq_length = 3;
  bool bidirectional = true;

  auto in = Variable(
      af::randu(input_size, batch_size, seq_length, af::dtype::f64), true);
  size_t n_params;

  switch (mode) {
    case RnnMode::TANH:
      n_params = 56;
      break;
    case RnnMode::LSTM:
      n_params = 224;
      break;
    case RnnMode::GRU:
      n_params = 168;
      break;
    default:
      throw std::invalid_argument("invalid RNN mode for the test");
  }

  auto w = Variable(af::randu(n_params, af::dtype::f64), true);

  auto func_rnn_in = [&](Variable& input) -> Variable {
    return std::get<0>(
        rnn(input,
            Variable(),
            Variable(),
            w,
            hidden_size,
            num_layers,
            mode,
            bidirectional,
            0.0));
  };
  ASSERT_TRUE(jacobianTestImpl(func_rnn_in, in, 1E-5));

  auto func_rnn_w = [&](Variable& weights) -> Variable {
    return std::get<0>(
        rnn(in,
            Variable(),
            Variable(),
            weights,
            hidden_size,
            num_layers,
            mode,
            bidirectional,
            0.0));
  };
  ASSERT_TRUE(jacobianTestImpl(func_rnn_w, w, 1E-5));

  // We get the correct gradient for hx
  auto hx = Variable(
      af::randu(
          input_size,
          batch_size,
          num_layers * (1 + bidirectional),
          af::dtype::f64),
      true);
  auto func_rnn_hx = [&](Variable& hidden_state) -> Variable {
    return std::get<0>(
        rnn(in,
            hidden_state,
            Variable(),
            w,
            hidden_size,
            num_layers,
            mode,
            bidirectional,
            0.0));
  };
  ASSERT_TRUE(jacobianTestImpl(func_rnn_hx, hx, 1E-5));

  // We can compute the gradient w.r.t. hy
  auto func_rnn_in_dhy = [&](Variable& input) -> Variable {
    return std::get<1>(
        rnn(input,
            Variable(),
            Variable(),
            w,
            hidden_size,
            num_layers,
            mode,
            bidirectional,
            0.0));
  };
  ASSERT_TRUE(jacobianTestImpl(func_rnn_in_dhy, in, 1E-5));

  if (mode == RnnMode::LSTM) {
    // We get the correct gradient for cx
    auto cx = Variable(
        af::randu(
            input_size,
            batch_size,
            num_layers * (1 + bidirectional),
            af::dtype::f64),
        true);
    auto func_rnn_cx = [&](Variable& cell_state) -> Variable {
      return std::get<0>(
          rnn(in,
              Variable(),
              cell_state,
              w,
              hidden_size,
              num_layers,
              mode,
              bidirectional,
              0.0));
    };
    ASSERT_TRUE(jacobianTestImpl(func_rnn_cx, cx, 1E-5));

    // We can compute the gradient w.r.t. cy
    auto func_rnn_in_dcy = [&](Variable& input) -> Variable {
      return std::get<2>(
          rnn(input,
              Variable(),
              Variable(),
              w,
              hidden_size,
              num_layers,
              mode,
              bidirectional,
              0.0));
    };
    ASSERT_TRUE(jacobianTestImpl(func_rnn_in_dcy, in, 1E-5));
  }
}

TEST(AutogradTest, Rnn) {
  test_rnn_impl(RnnMode::TANH);
}

TEST(AutogradTest, Lstm) {
  test_rnn_impl(RnnMode::LSTM);
}

TEST(AutogradTest, Gru) {
  test_rnn_impl(RnnMode::GRU);
}

TEST(AutogradTest, Embedding) {
  int n_words = 10;
  auto input = Variable((af::randu(4, 2) * n_words).as(s32), false);
  auto weights = Variable(af::randn(4, n_words, f64), true);
  auto func_embed = [&](Variable& w) { return embedding(input, w); };
  ASSERT_TRUE(jacobianTestImpl(func_embed, weights, 1E-5));
}

TEST(AutogradTest, BatchnormEvalModeOutput) {
  int feat_dims = 12;
  std::vector<int> feat_axes = {0, 1};
  auto input = Variable(af::randu(4, 3, 10), true);
  auto running_mean = Variable(af::constant(0.0, feat_dims), true);
  auto running_var = Variable(af::constant(1.0, feat_dims), true);

  // Eval mode (mean = 0, var = 1) with no affine transform.
  auto out = (batchnorm(
      input,
      Variable(),
      Variable(),
      running_mean,
      running_var,
      feat_axes,
      false,
      0.9,
      1E-5));
  ASSERT_TRUE(allClose(out.array(), input.array()));
  ASSERT_EQ(out.type(), input.type());

  // Eval mode (mean = 0, var = 1) with  affine transform.
  auto weight = Variable(af::randu(feat_dims), true);
  auto bias = Variable(af::randu(feat_dims), true);
  out = (batchnorm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      feat_axes,
      false,
      0.9,
      1E-5));
  for (int i = 0; i < feat_dims; ++i) {
    auto wt = weight.array()(i).scalar<float>();
    auto bs = bias.array()(i).scalar<float>();

    std::array<af::index, 4> sel = {af::span, af::span, af::span, af::span};
    sel[0] = i % input.dims(0);
    sel[1] = i / input.dims(0);

    auto expected_out = wt * input.array()(sel[0], sel[1], sel[2], sel[3]) + bs;
    ASSERT_TRUE(allClose(
        out.array()(sel[0], sel[1], sel[2], sel[3]), expected_out, 1E-5));
  }
}

TEST(AutogradTest, BatchnormTrainModeOutput) {
  int num_feat = 3;
  std::vector<int> feat_axes = {2};
  double epsilon = 1E-5;
  auto input = Variable(af::randu(4, 4, 3, 10), true);
  auto weight = Variable(af::randu(num_feat), true);
  auto bias = Variable(af::randu(num_feat), true);
  auto running = Variable();
  auto out = batchnorm(
      input, weight, bias, running, running, feat_axes, true, 0.9, epsilon);

  // Calculate maually
  auto todim = af::dim4(1, 1, num_feat);
  std::vector<int> nrm_axes = {0, 1, 3};
  auto avg = moddims(mean(input, nrm_axes), todim);
  auto variance =
      moddims(var(input, nrm_axes, true /* population var */), todim);
  auto expected_out = (input - tileAs(avg, input)) /
      fl::sqrt(tileAs(variance, input) + epsilon);
  expected_out = expected_out * tileAs(moddims(weight, todim), input) +
      tileAs(moddims(bias, todim), input);

  ASSERT_TRUE(allClose(out.array(), expected_out.array(), 1E-5));
}

TEST(AutogradTest, BatchnormJacobian) {
  af::dim4 feat_dims(6);
  std::vector<int> feat_axes = {1};
  auto input = Variable(af::randu(3, 6, 4, af::dtype::f64), true);
  auto running_mean = Variable(af::randu(feat_dims, af::dtype::f64), false);
  auto running_var = Variable(af::randu(feat_dims, af::dtype::f64), false);
  auto weight = Variable(af::randu(feat_dims, af::dtype::f64), true);
  auto bias = Variable(af::randu(feat_dims, af::dtype::f64), true);

  // Jacobian Test with  train_mode = true;

  auto func_bn_in = [&](Variable& in) {
    return (batchnorm(
        in,
        weight,
        bias,
        running_mean,
        running_var,
        feat_axes,
        true,
        0.9,
        1E-5));
  };

  ASSERT_TRUE(jacobianTestImpl(func_bn_in, input, 1E-2, 1E-4));

  auto func_bn_wt = [&](Variable& wt) {
    return (batchnorm(
        input,
        wt,
        bias,
        running_mean,
        running_var,
        feat_axes,
        true,
        0.9,
        1E-5));
  };

  ASSERT_TRUE(jacobianTestImpl(func_bn_wt, weight, 1E-4));

  auto func_bn_bs = [&](Variable& bs) {
    return (batchnorm(
        input,
        weight,
        bs,
        running_mean,
        running_var,
        feat_axes,
        true,
        0.9,
        1E-5));
  };

  ASSERT_TRUE(jacobianTestImpl(func_bn_bs, bias, 1E-4));
}

TEST(AutogradTest, LayerNormJacobian) {
  double eps = 1E-5;
  std::vector<int> feat_axes = {1};
  auto input = Variable(af::randu(4, 3, 10), true);

  Variable dummy_in_mean, dummy_in_var;

  auto func_ln_in = [&](Variable& in) {
    return batchnorm(
        in,
        Variable(),
        Variable(),
        dummy_in_mean,
        dummy_in_var,
        feat_axes,
        true,
        0.0,
        1E-5);
  };

  ASSERT_TRUE(jacobianTestImpl(func_ln_in, input, 1E-2, 1E-4));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
