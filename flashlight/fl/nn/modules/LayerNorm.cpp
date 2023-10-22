/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/LayerNorm.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/tensor/Shape.h"

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
  for (int d = 0; d < kLnExpectedNumDims; ++d) {
    if (std::find(axis.begin(), axis.end(), d) == axis.end()) {
      axisComplement_.push_back(d);
    }
  }
  initialize();
}

Variable LayerNorm::forward(const Variable& _input) {
  Variable input = _input;
  // If the input isn't of kLnExpectedNumDims, reshape so it is -- do this by
  // adding singleton dims. This is needed per computing the axis complement
  // TODO: this is pretty ugly -- eventually fix this up if it can be avoided
  if (input.ndim() < kLnExpectedNumDims) {
    std::vector<Dim> s = _input.shape().get();
    for (unsigned i = s.size(); i < kLnExpectedNumDims; ++i) {
      s.push_back(1);
    }
    input = moddims(_input, Shape(s));
  } else if (input.ndim() > kLnExpectedNumDims) {
    throw std::invalid_argument(
        "LayerNorm::forward - input must be " +
        std::to_string(kLnExpectedNumDims) + " or fewer dimensions.");
  }

  Variable dummyInMean, dummyInVar;

  Variable inputToBn = input;
  std::vector<int> inNormAxes;
  // reorder is only required if axisComplement_ is not continuous
  Shape reorderDims(std::vector<Dim>(input.ndim()));
  auto maxAxis =
      *std::max_element(axisComplement_.begin(), axisComplement_.end());
  auto minAxis =
      *std::min_element(axisComplement_.begin(), axisComplement_.end());
  bool axesContinuous = (axisComplement_.size() == (maxAxis - minAxis + 1));
  if (axesContinuous) {
    inNormAxes = axisComplement_;
  } else {
    int i = 0;
    for (int d = 0; d < input.ndim(); ++d) {
      if (std::find(axisComplement_.begin(), axisComplement_.end(), d) ==
          axisComplement_.end()) {
        reorderDims[i++] = d;
      }
    }
    for (auto n : axisComplement_) {
      inNormAxes.push_back(i);
      reorderDims[i++] = n;
    }
    inputToBn = reorder(input, reorderDims);
  }
  auto paramsType =
      (input.type() == fl::dtype::f16) ? fl::dtype::f32 : input.type();
  auto output = batchnorm(
      inputToBn,
      Variable(Tensor(paramsType), false),
      Variable(Tensor(paramsType), false),
      dummyInMean,
      dummyInVar,
      inNormAxes,
      true,
      0.0,
      epsilon_);

  if (!axesContinuous) {
    std::vector<std::pair<int, int>> restoreDims;
    for (size_t i = 0; i < reorderDims.ndim(); ++i) {
      restoreDims.emplace_back(reorderDims[i], i);
    }
    std::sort(restoreDims.begin(), restoreDims.end());
    Shape restoreDimsShape(std::vector<Dim>(restoreDims.size()));
    for (size_t i = 0; i < restoreDims.size(); ++i) {
      restoreDimsShape[i] = restoreDims[i].second;
    }
    output = reorder(output, restoreDimsShape);
  }

  if (affine_) {
    Variable weight = params_[0].astype(output.type());
    Variable bias = params_[1].astype(output.type());
    if (axisSize_ != kLnVariableAxisSize) {
      Shape affineDims = input.shape();
      for (int ax : axisComplement_) {
        affineDims[ax] = 1;
      }
      if (affineDims.elements() != axisSize_) {
        throw std::invalid_argument(
            "[LayerNorm] Input size along the norm axis doesn't with axisSize.");
      }
      weight = moddims(params_[0].astype(output.type()), affineDims);
      bias = moddims(params_[1].astype(output.type()), affineDims);
    }
    output = tileAs(weight, input) * output + tileAs(bias, input);
  }

  return moddims(output, _input.shape());
}

void LayerNorm::initialize() {
  if (affine_) {
    auto paramDim = (axisSize_ == kLnVariableAxisSize) ? 1 : axisSize_;
    auto wt = constant(1.0, {paramDim}, fl::dtype::f32, true);
    auto bs = constant(0.0, {paramDim}, fl::dtype::f32, true);
    params_ = {wt, bs};
  }
}

std::unique_ptr<Module> LayerNorm::clone() const {
  return std::make_unique<LayerNorm>(*this);
}

std::string LayerNorm::prettyString() const {
  std::ostringstream ss;
  ss << "LayerNorm";
  ss << " ( axis : { ";
  for (int d = 0; d < axisComplement_.size(); ++d) {
    if (std::find(axisComplement_.begin(), axisComplement_.end(), d) ==
        axisComplement_.end()) {
      ss << d << " ";
    }
  }
  ss << "} , size : " << axisSize_ << ")";
  return ss.str();
}

} // namespace fl
