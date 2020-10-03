/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/flashlight/nn/modules/BatchNorm.h"

#include "flashlight/flashlight/autograd/Functions.h"
#include "flashlight/flashlight/nn/Init.h"

namespace fl {

BatchNorm::BatchNorm(
    int featAxis,
    int featSize,
    double momentum /* = 0.1 */,
    double eps /*  = 1e-5*/,
    bool affine /*  = true*/,
    bool trackStats /*  = true*/)
    : BatchNorm(
          std::vector<int>(1, featAxis),
          featSize,
          momentum,
          eps,
          affine,
          trackStats) {}

BatchNorm::BatchNorm(
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

Variable BatchNorm::forward(const Variable& input) {
  double avgFactor = 0.0;

  if (train_ && trackStats_) {
    ++numBatchesTracked_;
    if (momentum_ < 0) { // cumulative moving average
      avgFactor = 1.0 / numBatchesTracked_;
    } else { // exponential moving average
      avgFactor = momentum_;
    }
  }

  return batchnorm(
      input,
      params_.empty() ? Variable() : params_[0],
      params_.empty() ? Variable() : params_[1],
      runningMean_,
      runningVar_,
      featAxis_,
      train_ || (!trackStats_),
      avgFactor,
      epsilon_);
}

void BatchNorm::initialize() {
  if (trackStats_) {
    runningMean_ = constant(0.0, featSize_, af::dtype::f32, false);
    runningVar_ = constant(1.0, featSize_, af::dtype::f32, false);
  }

  if (affine_) {
    auto wt = uniform(featSize_, 0.0, 1.0, af::dtype::f32, true);
    auto bs = constant(0.0, featSize_, af::dtype::f32, true);
    params_ = {wt, bs};
  }
}

void BatchNorm::setRunningMean(Variable v) {
  runningMean_ = v;
}

void BatchNorm::setRunningVar(Variable v) {
  runningVar_ = v;
}

Variable BatchNorm::getRunningMean() {
  return runningMean_;
}

Variable BatchNorm::getRunningVar() {
  return runningVar_;
}

std::string BatchNorm::prettyString() const {
  std::ostringstream ss;
  ss << "BatchNorm";
  ss << " ( axis : { ";
  for (auto x : featAxis_) {
    ss << x << " ";
  }
  ss << "}, size : " << featSize_ << " )";
  return ss.str();
}

} // namespace fl
