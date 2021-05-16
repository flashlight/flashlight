/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/nn/FrozenBatchNorm.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"

namespace fl {

FrozenBatchNorm::FrozenBatchNorm(
    int featAxis,
    int featSize,
    double momentum /* = 0.1 */,
    double eps /*  = 1e-5*/,
    bool affine /*  = true*/,
    bool trackStats /*  = true*/)
    : FrozenBatchNorm(
          std::vector<int>(1, featAxis),
          featSize,
          momentum,
          eps,
          affine,
          trackStats) {}

FrozenBatchNorm::FrozenBatchNorm(
    const std::vector<int>& featAxis,
    int featSize,
    double momentum /* = 0.1*/,
    double eps /* = 1e-5 */,
    bool affine /* = true*/,
    bool trackStats /* = true*/)
    : BatchNorm(featAxis, featSize, momentum, eps, affine, trackStats) {
  BatchNorm::initialize();
}

Variable FrozenBatchNorm::forward(const Variable& input) {
  auto scale = params_[0] / fl::sqrt(runningVar_ + epsilon_);
  auto bias = params_[1] - runningMean_ * scale;
  bias = fl::moddims(bias, {1, 1, bias.dims(0), 1}).as(input.type());
  scale = fl::moddims(scale, {1, 1, scale.dims(0), 1}).as(input.type());
  return (input * fl::tileAs(scale, input)) + fl::tileAs(bias, input);
}

void FrozenBatchNorm::setRunningMean(const fl::Variable& x) {
  runningMean_ = x;
}

void FrozenBatchNorm::setRunningVar(const fl::Variable& x) {
  runningVar_ = x;
}

void FrozenBatchNorm::train() {
  for (auto& param : params_) {
    param.setCalcGrad(false);
  }
  runningVar_.setCalcGrad(false);
  runningMean_.setCalcGrad(false);
  train_ = false;
}

std::string FrozenBatchNorm::prettyString() const {
  std::ostringstream ss;
  ss << "FrozenBatchNorm";
  ss << " ( axis : { ";
  for (auto x : featAxis_) {
    ss << x << " ";
  }
  ss << "}, size : " << featSize_ << " )";
  return ss.str();
}

} // namespace fl
