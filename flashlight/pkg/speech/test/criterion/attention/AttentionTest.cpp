/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/flashlight.h"

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/pkg/speech/criterion/attention/Utils.h"
#include "flashlight/pkg/speech/criterion/attention/attention.h"

using namespace fl;
using namespace fl::pkg::speech;

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

void sequentialTest(std::shared_ptr<AttentionBase> attention, int H) {
  int B = 2, T = 10;

  Variable encodedx(fl::randn({H, T, B}), true);
  Variable encodedy(fl::randn({H, 1, B}), true);

  Variable alphas, summaries;
  for (int step = 0; step < 3; ++step) {
    std::tie(alphas, summaries) =
        attention->forward(encodedy, encodedx, alphas);
    ASSERT_EQ(alphas.shape(), Shape({1, T, B}));
    ASSERT_EQ(summaries.shape(), Shape({H, 1, B}));

    auto alphasum = fl::sum(alphas.tensor(), {1});
    auto ones = fl::full(alphasum.shape(), 1.0, alphasum.type());
    ASSERT_TRUE(allClose(alphasum, ones, 1e-5));
  }

  Variable windowMask = Variable(fl::full({1, T, B}, 0.), false);
  auto alphas1 =
      std::get<0>(attention->forward(encodedy, encodedx, alphas, windowMask));
  auto alphas2 = std::get<0>(attention->forward(encodedy, encodedx, alphas));
  ASSERT_TRUE(allClose(alphas1, alphas2, 1e-6));

  Variable encodedyInvalid(fl::randn({H, 10, B}), true);
  EXPECT_THROW(
      attention->forward(encodedyInvalid, encodedx, alphas),
      std::invalid_argument);
}

void sequentialTestWithPad(std::shared_ptr<AttentionBase> attention, int H) {
  int B = 2, T = 10;

  Variable encodedx(fl::randn({H, T, B}), true);
  std::vector<int> padRaw = {T / 2, T};
  Variable pad = Variable(Tensor::fromVector({1, B}, padRaw), false);
  Variable encodedy(fl::randn({H, 1, B}), true);

  Variable alphas, summaries;
  for (int step = 0; step < 3; ++step) {
    std::tie(alphas, summaries) =
        attention->forward(encodedy, encodedx, alphas, Variable(), pad);
    ASSERT_EQ(alphas.shape(), Shape({1, T, B}));
    ASSERT_EQ(summaries.shape(), Shape({H, 1, B}));

    auto alphasum = fl::sum(alphas.tensor(), {1});
    auto ones = fl::full(alphasum.shape(), 1.0, alphasum.type());
    ASSERT_TRUE(allClose(alphasum, ones, 1e-5));
    ASSERT_EQ(
        fl::countNonzero(
            alphas.tensor()(fl::span, fl::range(T - T / 2, T), 0) == 0)
            .scalar<unsigned>(),
        T / 2);
  }

  Variable windowMask = Variable(fl::full({1, T, B}, 0.0), false);
  auto alphas1 = std::get<0>(
      attention->forward(encodedy, encodedx, alphas, windowMask, pad));
  auto alphas2 = std::get<0>(
      attention->forward(encodedy, encodedx, alphas, Variable{}, pad));
  ASSERT_TRUE(allClose(alphas1, alphas2, 1e-6));

  Variable encodedyInvalid(fl::randn({H, 10, B}), true);
  EXPECT_THROW(
      attention->forward(encodedyInvalid, encodedx, alphas),
      std::invalid_argument);
}

} // namespace

TEST(AttentionTest, NeuralContentAttention) {
  int H = 8, B = 2, T = 10, U = 5;
  NeuralContentAttention attention(H);

  Variable encodedx(fl::randn({H, T, B}), true);
  Variable encodedy(fl::randn({H, U, B}), true);

  std::vector<int> padRaw = {T / 2, T};
  Variable pad = Variable(Tensor::fromVector(Shape({1, B}), padRaw), false);

  std::vector<Variable> padV = {Variable(), pad};
  for (const auto& currentPad : padV) {
    Variable alphas, summaries;
    std::tie(alphas, summaries) = attention.forward(
        encodedy, encodedx, Variable{}, Variable{}, currentPad);
    ASSERT_EQ(alphas.shape(), Shape({U, T, B}));
    ASSERT_EQ(summaries.shape(), Shape({H, U, B}));
    if (!currentPad.isEmpty()) {
      ASSERT_EQ(
          fl::countNonzero(
              alphas.tensor()(fl::span, fl::range(T - T / 2, T), 0) == 0)
              .scalar<unsigned>(),
          T / 2 * U);
    }
    auto alphasum = sum(alphas.tensor(), {1});
    auto ones = fl::full(alphasum.shape(), 1.0, alphasum.type());
    ASSERT_TRUE(allClose(alphasum, ones, 1e-5));

    Variable windowMask = Variable(fl::full({U, T, B}, 0.0), false);
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
  Variable pad = Variable(Tensor::fromVector({1, B}, padRaw), false);

  std::vector<Variable> padV = {Variable(), pad};
  for (const auto& currentPad : padV) {
    for (bool keyValue : {true, false}) {
      for (bool splitInput : {true, false}) {
        MultiHeadContentAttention attention(H, NH, keyValue, splitInput);

        auto hEncode = keyValue ? H * 2 : H;
        Variable encodedx(fl::randn({hEncode, T, B}), true);
        Variable encodedy(fl::randn({H, U, B}), true);

        Variable alphas, summaries;
        std::tie(alphas, summaries) = attention.forward(
            encodedy, encodedx, Variable{}, Variable{}, currentPad);
        ASSERT_EQ(alphas.shape(), Shape({U * NH, T, B}));
        ASSERT_EQ(summaries.shape(), Shape({H, U, B}));
        if (!currentPad.isEmpty()) {
          ASSERT_EQ(
              fl::countNonzero(
                  alphas.tensor()(fl::span, fl::range(T - T / 2, T), 0) == 0)
                  .scalar<unsigned>(),
              T / 2 * U * NH);
        }

        auto alphasum = sum(alphas.tensor(), {1});
        auto ones = fl::full(alphasum.shape(), 1.0, alphasum.type());
        ASSERT_TRUE(allClose(alphasum, ones, 1e-5));
      }
    }
  }
}

TEST(AttentionTest, JacobianMaskAttention) {
  // CxTxB
  auto in = Variable(fl::rand({10, 9, 5}, fl::dtype::f32), true);
  std::vector<int> inpSzRaw = {1, 2, 4, 8, 16};
  Tensor inpSz = Tensor::fromVector(
      {1, static_cast<long long>(inpSzRaw.size())}, inpSzRaw);
  auto func_in = [&](Variable& input) {
    return fl::pkg::speech::maskAttention(input, fl::Variable(inpSz, false));
  };
  ASSERT_TRUE(jacobianTestImpl(func_in, in, 2e-4));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
