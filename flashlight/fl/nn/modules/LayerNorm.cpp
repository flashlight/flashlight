/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/LayerNorm.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"

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
  Variable dummyInMean, dummyInVar;

  Variable inputToBn = input;
  std::vector<int> inNormAxes;
  // reorder is only required if axisComplement_ is not continuous
  std::array<int, AF_MAX_DIMS> reorderDims;
  auto maxAxis =
      *std::max_element(axisComplement_.begin(), axisComplement_.end());
  auto minAxis =
      *std::min_element(axisComplement_.begin(), axisComplement_.end());
  bool axesContinuous = (axisComplement_.size() == (maxAxis - minAxis + 1));
  if (axesContinuous) {
    inNormAxes = axisComplement_;
  } else {
    int i = 0;
    for (int d = 0; d < AF_MAX_DIMS; ++d) {
      if (std::find(axisComplement_.begin(), axisComplement_.end(), d) ==
          axisComplement_.end()) {
        reorderDims[i++] = d;
      }
    }
    for (auto n : axisComplement_) {
      inNormAxes.push_back(i);
      reorderDims[i++] = n;
    }
    inputToBn = reorder(
        input, reorderDims[0], reorderDims[1], reorderDims[2], reorderDims[3]);
  }
  auto paramsType =
      (input.type() == af::dtype::f16) ? af::dtype::f32 : input.type();
  auto output = batchnorm(
      inputToBn,
      Variable(af::array().as(paramsType), false),
      Variable(af::array().as(paramsType), false),
      dummyInMean,
      dummyInVar,
      inNormAxes,
      true,
      0.0,
      epsilon_);

  if (!axesContinuous) {
    std::vector<std::pair<int, int>> restoreDims = {{reorderDims[0], 0},
                                                    {reorderDims[1], 1},
                                                    {reorderDims[2], 2},
                                                    {reorderDims[3], 3}};
    std::sort(restoreDims.begin(), restoreDims.end());
    output = reorder(
        output,
        restoreDims[0].second,
        restoreDims[1].second,
        restoreDims[2].second,
        restoreDims[3].second);
  }

  if (affine_) {
    Variable weight = params_[0].as(output.type());
    Variable bias = params_[1].as(output.type());
    if (axisSize_ != kLnVariableAxisSize) {
      af::dim4 affineDims = input.dims();
      for (int ax : axisComplement_) {
        affineDims[ax] = 1;
      }
      if (affineDims.elements() != axisSize_) {
        throw std::invalid_argument(
            "[LayerNorm] Input size along the norm axis doesn't with axisSize.");
      }
      weight = moddims(params_[0].as(output.type()), affineDims);
      bias = moddims(params_[1].as(output.type()), affineDims);
    }
    output = tileAs(weight, input) * output + tileAs(bias, input);
  }

  return output;
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
