/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gtest/gtest.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/autograd/Variable.h"

namespace fl {
namespace detail {

class AutogradTestF16 : public ::testing::Test {
    void SetUp() override {
        // Ensures all operations will be in f16
        OptimMode::get().setOptimLevel(OptimLevel::O3);
    }

    void TearDown() override {
        OptimMode::get().setOptimLevel(OptimLevel::DEFAULT);
    }
};

using JacobianFunc = std::function<Variable(Variable&)>;
inline bool jacobianTestImpl(
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

}
} // namespace fl
