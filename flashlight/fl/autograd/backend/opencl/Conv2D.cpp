/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <arrayfire.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"

namespace fl {

namespace detail {
class ConvBenchmark;
} // namespace detail

namespace {

// Input, output: WHCN; weights: WHIO
constexpr size_t kWIdx = 0;
constexpr size_t kHIdx = 1;
constexpr size_t kIOChannelSizeIdx = 2;
constexpr size_t kIOBatchSizeIdx = 3;
constexpr size_t kWeightOutputChannelSizeIdx = 3;

} // namespace

Variable conv2d(
    const Variable& input,
    const Variable& weights,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups,
    std::shared_ptr<detail::ConvBenchmarks> benchmarks) {
  if (input.type() == f16) {
    throw std::runtime_error("Half precision is not supported in opencl.");
  }
  auto dummy_bias = Variable(af::array(), false);
  return conv2d(input, weights, dummy_bias, sx, sy, px, py, dx, dy, groups);
}

Variable conv2d(
    const Variable& input,
    const Variable& weights,
    const Variable& bias,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups,
    std::shared_ptr<detail::ConvBenchmarks> benchmarks) {
  if (input.type() == f16) {
    throw std::runtime_error("Half precision is not supported in CPU.");
  }
  auto output = af::array(
      1 +
          (input.dims(kWIdx) + (2 * px) -
           (1 + (weights.dims(kWIdx) - 1) * dx)) /
              sx,
      1 +
          (input.dims(kHIdx) + (2 * py) -
           (1 + (weights.dims(kHIdx) - 1) * dy)) /
              sy,
      weights.dims(kWeightOutputChannelSizeIdx),
      input.dims(kIOBatchSizeIdx));
  auto hasBias = bias.elements() > 0;

  // flashlight input, weight, and output shapes in column-major:
  // - Input is WHCN
  // - Weights are WHIO
  // - Output is WHCN
  // Since ArrayFire is column major,...

  /*...*/

  /***************************** Backward ******************************/
  auto gradFunc = [hasBias /*...*/
  ](std::vector<Variable>& inputs, const Variable& grad_output) {
    auto& inputRef = inputs[0];
    auto& weightRef = inputs[1];

    /*...*/

    /*** Compute the gradient with respect to the input ***/
    if (inputRef.isCalcGrad()) {
      // Result
      auto gradInput = Variable(/*...*/);
      /*...*/

      inputRef.addGrad(gradInput);
    }

    /*** Compute the gradient with respect to weight and bias ***/
    if (weightRef.isCalcGrad()) {
      // Result
      auto gradWeights = Variable(/*...*/);
      Variable gradBias;
      bool computeBiasGrad = hasBias && inputs[2].isCalcGrad();
      auto& biasRef = inputs[2];
      if (computeBiasGrad) {
        gradBias = Variable(/*...*/);
      }
      /*...*/

      // Add weight and bias gradients
      weightRef.addGrad(gradWeights);
      if (computeBiasGrad) {
        biasRef.addGrad(gradBias);
      }
    }
  };

  throw std::runtime_error("conv2d not yet implemented on opencl");

  // Return for forward
  if (hasBias) {
    return Variable(output, {input, weights, bias}, gradFunc);
  }
  return Variable(output, {input, weights}, gradFunc);
}

} // namespace fl
