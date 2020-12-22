/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
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
constexpr size_t kWeightOutputChannelSizeIdx = 3;

Variable conv2dWithoutBiasAndGroups(
    const Variable& input,
    const Variable& weights,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int deviceId) {
  if (input.dims(kIOChannelSizeIdx) != weights.dims(kIOChannelSizeIdx)) {
    throw std::runtime_error(
        "input(kIOChannelSizeIdx) != weights(kIOChannelSizeIdx)");
  }

  const auto stride = af::dim4(sx, sy);
  const auto padding = af::dim4(std::max(px, 1), std::max(py, 1));
  const auto dilation = af::dim4(dx, dy);
  auto weightsFlip = af::flip(af::flip(weights.array(), 0), 1);

  auto convOut =
      af::convolve2NN(input.array(), weightsFlip, stride, padding, dilation);

  auto gradFunc = [stride, padding, dilation, convOut, deviceId, weightsFlip](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    af::setDevice(deviceId);

    auto& in = inputs[0];
    auto& wt = inputs[1];

    if (in.isCalcGrad()) {
      auto inGrad = convolve2GradientNN(
          gradOutput.array(),
          in.array(),
          weightsFlip,
          convOut,
          stride,
          padding,
          dilation,
          AF_CONV_GRADIENT_DATA);
      in.addGrad(fl::Variable(inGrad, false));
    }
    if (wt.isCalcGrad()) {
      auto wtGrad = convolve2GradientNN(
          gradOutput.array(),
          in.array(),
          weightsFlip,
          convOut,
          stride,
          padding,
          dilation,
          AF_CONV_GRADIENT_FILTER);
      // wt are weights are in original orientation but wtGrad is flipped so we
      // flip wtGrad before adding as a grad to wt.
      wt.addGrad(fl::Variable(af::flip(af::flip(wtGrad, 0), 1), false));
    };
  };
  return Variable(convOut, {input, weights}, gradFunc);
}

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
  Variable dummy_bias = Variable(af::array(), false);
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
    throw std::runtime_error("Half precision is not supported in opencl.");
  }
  const int chan = input.dims(kIOChannelSizeIdx);
  if ((chan % groups) != 0) {
    throw std::runtime_error(
        "Number of channels must be devisible by number of groups");
  }
  auto deviceId = af::getDevice();

  std::vector<Variable> groupInput = fl::split(
      input, input.dims(kIOChannelSizeIdx) / groups, kIOChannelSizeIdx);
  std::vector<Variable> groupWeights = fl::split(
      weights,
      weights.dims(kWeightOutputChannelSizeIdx) / groups,
      kWeightOutputChannelSizeIdx);
  std::vector<Variable> groupOutput(groups);

  if (groupInput.size() != groupWeights.size() ||
      groupInput.size() != groupOutput.size()) {
    throw std::runtime_error(
        "Number of groups must match for input, weights, and output");
  }

  for (int g = 0; g < groups; ++g) {
    groupOutput[g] = conv2dWithoutBiasAndGroups(
        groupInput[g], groupWeights[g], sx, sy, px, py, dx, dy, deviceId);
  }

  auto output = fl::concatenate(groupOutput, kIOChannelSizeIdx);

  if (!bias.isempty()) {
    auto tiledBias = fl::tileAs(bias, output);
    output = output + tiledBias;
  }

  // ArrayFire convolve2NN padding must be at least 1. Trim when padding
  // is zero.
  if (px >= 1 && py >= 1) {
    return output;
  } else {
    const int inX = input.dims(kWIdx);
    const int wtX = weights.dims(kWIdx);
    const int outWantX = 1 + (inX + 2 * px - (1 + (wtX - 1) * dx)) / sx;
    int outHaveX = output.dims(kWIdx);
    int outFirstX = (outHaveX - outWantX) / 2;
    af::seq seqX(outFirstX, outFirstX + outWantX - 1);

    const int inY = input.dims(kHIdx);
    const int wtY = weights.dims(kHIdx);
    const int outWantY = 1 + (inY + 2 * py - (1 + (wtY - 1) * dy)) / sy;
    int outHaveY = output.dims(kHIdx);
    int outFirstY = (outHaveY - outWantY) / 2;
    af::seq seqY(outFirstY, outFirstY + outWantY - 1);

    return output(seqX, seqY, af::span, af::span);
  }
}

} // namespace fl
