/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/image/fl/models/FrozenBatchNorm.h"

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
    : featAxis_(featAxis),
      featSize_(featSize),
      numBatchesTracked_(0),
      momentum_(momentum),
      epsilon_(eps),
      affine_(affine),
      trackStats_(trackStats) {
  initialize();
}


Variable FrozenBatchNorm::forward(const Variable& input) {
  auto scale = params_[0] / fl::sqrt(runningVar_ + epsilon_);
  auto bias = params_[1] - runningMean_ * scale;
  bias = fl::moddims(bias, { 1, 1, bias.dims(0), 1});
  scale = fl::moddims(scale, { 1, 1, scale.dims(0), 1});
  return (input * fl::tileAs(scale, input)) + fl::tileAs(bias, input);
}

void FrozenBatchNorm::initialize() {
  if (trackStats_) {
    runningMean_ = constant(0.0, featSize_, af::dtype::f32, false);
    runningVar_ = constant(1.0, featSize_, af::dtype::f32, false);
  }

  if (affine_) {
    auto wt = uniform(featSize_, 0.0, 1.0, af::dtype::f32, false);
    auto bs = constant(0.0, featSize_, af::dtype::f32, false);
    params_ = {wt, bs};
  }
}

void FrozenBatchNorm::setRunningMean(af::array x) {
  runningMean_ = fl::Variable(x, false);
}

void FrozenBatchNorm::setRunningVar(af::array x) {
  runningVar_ = fl::Variable(x, false);
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
