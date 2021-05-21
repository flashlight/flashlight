/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <arrayfire.h>
#include "flashlight/fl/flashlight.h"

#include "flashlight/pkg/speech/criterion/attention/Utils.h"
#include "flashlight/pkg/speech/criterion/attention/attention.h"

using namespace fl;
using namespace fl::app::asr;

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

void sequentialTest(std::shared_ptr<AttentionBase> attention, int H) {
  int B = 2, T = 10;

  Variable encodedx(af::randn(H, T, B), true);
  Variable encodedy(af::randn(H, 1, B), true);

  Variable alphas, summaries;
  for (int step = 0; step < 3; ++step) {
    std::tie(alphas, summaries) =
        attention->forward(encodedy, encodedx, alphas);
    ASSERT_EQ(alphas.dims(), af::dim4(1, T, B));
    ASSERT_EQ(summaries.dims(), af::dim4(H, 1, B));

    auto alphasum = af::sum(alphas.array(), 1);
    auto ones = af::constant(1.0, alphasum.dims(), alphasum.type());
    ASSERT_TRUE(allClose(alphasum, ones, 1e-5));
  }

  Variable windowMask = Variable(af::constant(0.0, 1, T, B), false);
  auto alphas1 =
      std::get<0>(attention->forward(encodedy, encodedx, alphas, windowMask));
  auto alphas2 = std::get<0>(attention->forward(encodedy, encodedx, alphas));
  ASSERT_TRUE(allClose(alphas1, alphas2, 1e-6));

  Variable encodedyInvalid(af::randn(H, 10, B), true);
  EXPECT_THROW(
      attention->forward(encodedyInvalid, encodedx, alphas),
      std::invalid_argument);
}

void sequentialTestWithPad(std::shared_ptr<AttentionBase> attention, int H) {
  int B = 2, T = 10;

  Variable encodedx(af::randn(H, T, B), true);
  std::vector<int> padRaw = {T / 2, T};
  Variable pad = Variable(af::array(af::dim4(1, B), padRaw.data()), false);
  Variable encodedy(af::randn(H, 1, B), true);

  Variable alphas, summaries;
  for (int step = 0; step < 3; ++step) {
    std::tie(alphas, summaries) =
        attention->forward(encodedy, encodedx, alphas, Variable(), pad);
    ASSERT_EQ(alphas.dims(), af::dim4(1, T, B));
    ASSERT_EQ(summaries.dims(), af::dim4(H, 1, B));

    auto alphasum = af::sum(alphas.array(), 1);
    auto ones = af::constant(1.0, alphasum.dims(), alphasum.type());
    ASSERT_TRUE(allClose(alphasum, ones, 1e-5));
    ASSERT_TRUE(
        af::count<int>(
            alphas.array()(af::span, af::seq(T - T / 2, T - 1), 0) == 0) ==
        T / 2);
  }

  Variable windowMask = Variable(af::constant(0.0, 1, T, B), false);
  auto alphas1 = std::get<0>(
      attention->forward(encodedy, encodedx, alphas, windowMask, pad));
  auto alphas2 = std::get<0>(
      attention->forward(encodedy, encodedx, alphas, Variable{}, pad));
  ASSERT_TRUE(allClose(alphas1, alphas2, 1e-6));

  Variable encodedyInvalid(af::randn(H, 10, B), true);
  EXPECT_THROW(
      attention->forward(encodedyInvalid, encodedx, alphas),
      std::invalid_argument);
}

} // namespace

TEST(AttentionTest, NeuralContentAttention) {
  int H = 8, B = 2, T = 10, U = 5;
  NeuralContentAttention attention(H);

  Variable encodedx(af::randn(H, T, B), true);
  Variable encodedy(af::randn(H, U, B), true);

  std::vector<int> padRaw = {T / 2, T};
  Variable pad = Variable(af::array(af::dim4(1, B), padRaw.data()), false);

  std::vector<Variable> padV = {Variable(), pad};
  for (auto currentPad : padV) {
    Variable alphas, summaries;
    std::tie(alphas, summaries) = attention.forward(
        encodedy, encodedx, Variable{}, Variable{}, currentPad);
    ASSERT_EQ(alphas.dims(), af::dim4(U, T, B));
    ASSERT_EQ(summaries.dims(), af::dim4(H, U, B));
    if (!currentPad.isempty()) {
      ASSERT_TRUE(
          af::count<int>(
              alphas.array()(af::span, af::seq(T - T / 2, T - 1), 0) == 0) ==
          T / 2 * U);
    }
    auto alphasum = sum(alphas.array(), 1);
    auto ones = af::constant(1.0, alphasum.dims(), alphasum.type());
    ASSERT_TRUE(allClose(alphasum, ones, 1e-5));

    Variable windowMask = Variable(af::constant(0.0, U, T, B), false);
    auto alphas1 = std::get<0>(attention.forward(
        encodedy, encodedx, Variable{}, windowMask, currentPad));
    ASSERT_TRUE(allClose(alphas, alphas1, 1e-6));
  }
}

TEST(AttentionTest, SimpleLocationAttention) {
  int H = 8, K = 5;
  sequentialTest(std::make_shared<SimpleLocationAttention>(K), H);
  sequentialTestWithPad(std::make_shared<SimpleLocationAttention>(K), H);
}

TEST(AttentionTest, LocationAttention) {
  int H = 8, K = 5;
  sequentialTest(std::make_shared<LocationAttention>(H, K), H);
  sequentialTestWithPad(std::make_shared<LocationAttention>(H, K), H);
}

TEST(AttentionTest, NeuralLocationAttention) {
  int H = 8, A = 8, C = 5, K = 3;
  sequentialTest(std::make_shared<NeuralLocationAttention>(H, A, C, K), H);
  sequentialTestWithPad(
      std::make_shared<NeuralLocationAttention>(H, A, C, K), H);
}

TEST(AttentionTest, MultiHeadContentAttention) {
  int H = 512, B = 2, T = 10, U = 5, NH = 8;

  std::vector<int> padRaw = {T / 2, T};
  Variable pad = Variable(af::array(af::dim4(1, B), padRaw.data()), false);

  std::vector<Variable> padV = {Variable(), pad};
  for (auto currentPad : padV) {
    for (bool keyValue : {true, false}) {
      for (bool splitInput : {true, false}) {
        MultiHeadContentAttention attention(H, NH, keyValue, splitInput);

        auto hEncode = keyValue ? H * 2 : H;
        Variable encodedx(af::randn(hEncode, T, B), true);
        Variable encodedy(af::randn(H, U, B), true);

        Variable alphas, summaries;
        std::tie(alphas, summaries) = attention.forward(
            encodedy, encodedx, Variable{}, Variable{}, currentPad);
        ASSERT_EQ(alphas.dims(), af::dim4(U * NH, T, B));
        ASSERT_EQ(summaries.dims(), af::dim4(H, U, B));
        if (!currentPad.isempty()) {
          ASSERT_TRUE(
              af::count<int>(
                  alphas.array()(af::span, af::seq(T - T / 2, T - 1), 0) ==
                  0) == T / 2 * U * NH);
        }

        auto alphasum = sum(alphas.array(), 1);
        auto ones = af::constant(1.0, alphasum.dims(), alphasum.type());
        ASSERT_TRUE(allClose(alphasum, ones, 1e-5));
      }
    }
  }
}

TEST(AttentionTest, JacobianMaskAttention) {
  // CxTxB
  auto in = Variable(af::randu(10, 9, 5, af::dtype::f32), true);
  std::vector<int> inpSzRaw = {1, 2, 4, 8, 16};
  af::array inpSz = af::array(af::dim4(1, inpSzRaw.size()), inpSzRaw.data());
  auto func_in = [&](Variable& input) {
    return fl::app::asr::maskAttention(input, fl::Variable(inpSz, false));
  };
  ASSERT_TRUE(jacobianTestImpl(func_in, in, 2e-4));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
