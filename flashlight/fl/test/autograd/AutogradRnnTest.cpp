/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// TODO move all other Rnn tests to this file (e.g. from F16 tests)
#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"

using namespace ::testing;
using namespace fl;

using fl::detail::AutogradTestF16;

namespace {

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
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcRnnIn, in, expectedPrecision, perturbation));

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
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcRnnW, w, expectedPrecision, perturbation));

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
    ASSERT_TRUE(fl::detail::jacobianTestImpl(funcRnnHx, hx, expectedPrecision, perturbation));

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
        fl::detail::jacobianTestImpl(funcRnnInDhy, in, expectedPrecision, perturbation));

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
            fl::detail::jacobianTestImpl(funcRnnCx, cx, expectedPrecision, perturbation));

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
            fl::detail::jacobianTestImpl(funcRnnInDcy, in, expectedPrecision, perturbation));
    }
}

}

TEST(AutogradRnnTest, Rnn) {
  if (FL_BACKEND_CPU) {
    GTEST_SKIP() << "RNN gradient computation not yet supported on CPU";
  }

  testRnnImpl(RnnMode::TANH);
}

TEST(AutogradRnnTest, Lstm) {
  if (FL_BACKEND_CPU) {
    GTEST_SKIP() << "RNN LSTM graident computation not yet supported on CPU";
  }

  testRnnImpl(RnnMode::LSTM);
}

TEST(AutogradRnnTest, Gru) {
  if (FL_BACKEND_CPU) {
    GTEST_SKIP() << "RNN GRU graident computation not yet supported on CPU";
  }
  testRnnImpl(RnnMode::GRU);
}

TEST_F(AutogradTestF16, RnnF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  testRnnImpl(RnnMode::TANH, fl::dtype::f16);
}

TEST_F(AutogradTestF16, LstmF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  testRnnImpl(RnnMode::LSTM, fl::dtype::f16);
}

TEST_F(AutogradTestF16, GruF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  testRnnImpl(RnnMode::GRU, fl::dtype::f16);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
