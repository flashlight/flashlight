/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "BatchNorm.h"

#include <flashlight/autograd/Functions.h>
#include <flashlight/nn/Init.h>

namespace fl {

BatchNorm::BatchNorm(
    int feat_axis,
    intl feat_size,
    double momentum /* = 0.1 */,
    double eps /*  = 1e-5*/,
    bool affine /*  = true*/,
    bool track_stats /*  = true*/)
    : BatchNorm(
          std::vector<int>(1, feat_axis),
          feat_size,
          momentum,
          eps,
          affine,
          track_stats) {}

BatchNorm::BatchNorm(
    const std::vector<int>& feat_axes,
    intl feat_size,
    double momentum /* = 0.1*/,
    double eps /* = 1e-5 */,
    bool affine /* = true*/,
    bool track_stats /* = true*/)
    : featAxes_(feat_axes),
      featSize_(feat_size),
      numBatchesTracked_(0),
      momentum_(momentum),
      epsilon_(eps),
      affine_(affine),
      trackStats_(track_stats) {
  initialize();
}

Variable BatchNorm::forward(const Variable& input) {
  double avg_factor = 0.0;

  if (train_ && trackStats_) {
    ++numBatchesTracked_;
    if (momentum_ < 0) { // cumulative moving average
      avg_factor = 1.0 / numBatchesTracked_;
    } else { // exponential moving average
      avg_factor = momentum_;
    }
  }

  return batchnorm(
      input,
      params_.empty() ? Variable() : params_[0],
      params_.empty() ? Variable() : params_[1],
      runningMean_,
      runningVar_,
      featAxes_,
      train_ || (!trackStats_),
      avg_factor,
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

std::string BatchNorm::prettyString() const {
  std::ostringstream ss;
  ss << "BatchNorm";
  ss << " ( axes : { ";
  for (auto x : featAxes_) {
    ss << x << " ";
  }
  ss << "}, size : " << featSize_ << " )";
  return ss.str();
}

} // namespace fl
