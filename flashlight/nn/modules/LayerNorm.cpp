/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/nn/modules/LayerNorm.h"

#include <numeric>
#include <stdexcept>

#include "flashlight/autograd/Functions.h"
#include "flashlight/nn/Init.h"

namespace fl {

LayerNorm::LayerNorm(
    int axis,
    double eps /*  = 1e-5*/,
    bool affine /*  = true*/,
    int axisSize /* = kLnVariableAxisSize */)
    : LayerNorm(std::vector<int>({axis}), eps, affine, axisSize) {}

LayerNorm::LayerNorm(
    const std::vector<int>& axis,
    double eps /* = 1e-5 */,
    bool affine /* = true */,
    int axisSize /* = kLnVariableAxisSize */)
    : epsilon_(eps), affine_(affine), axisSize_(axisSize) {
  for (int d = 0; d < AF_MAX_DIMS; ++d) {
    if (std::find(axis.begin(), axis.end(), d) == axis.end()) {
      axisComplement_.push_back(d);
    }
  }
  initialize();
}

Variable LayerNorm::forward(const Variable& input) {
  std::vector<int> axis;
  for (int d = 0; d < AF_MAX_DIMS; ++d) {
    if (std::find(axisComplement_.begin(), axisComplement_.end(), d) ==
        axisComplement_.end()) {
      axis.push_back(d);
    }
  }

  auto mean = fl::tileAs(fl::mean(input, axis), input);
  auto stddev =
      fl::tileAs(fl::sqrt(fl::var(input, axis, true) + epsilon_), input);

  auto output = (input - mean) / stddev;

  if (!affine_) {
    return output;
  }
  Variable weight = params_[0], bias = params_[1];
  if (axisSize_ != kLnVariableAxisSize) {
    af::dim4 featDims(1, 1, 1, 1);
    for (auto i : axis) {
      featDims[i] = input.dims(i);
    }
    weight = fl::moddims(params_[0], featDims);
    bias = fl::moddims(params_[1], featDims);
  }
  return tileAs(weight, input) * output + tileAs(bias, input);
}

void LayerNorm::initialize() {
  if (affine_) {
    auto paramDim = (axisSize_ == kLnVariableAxisSize) ? 1 : axisSize_;
    auto wt = constant(1.0, paramDim, af::dtype::f32, true);
    auto bs = constant(0.0, paramDim, af::dtype::f32, true);
    params_ = {wt, bs};
  }
}

std::string LayerNorm::prettyString() const {
  std::ostringstream ss;
  ss << "LayerNorm";
  ss << " ( axis : { ";
  for (int d = 0; d < AF_MAX_DIMS; ++d) {
    if (std::find(axisComplement_.begin(), axisComplement_.end(), d) ==
        axisComplement_.end()) {
      ss << d << " ";
    }
  }
  ss << "} , size : " << axisSize_ << ")";
  return ss.str();
}

} // namespace fl
