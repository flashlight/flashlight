/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/optim/RMSPropOptimizer.h"

#include <cmath>

#include "flashlight/fl/tensor/Compute.h"

using std::vector;

namespace fl {

RMSPropOptimizer::RMSPropOptimizer(
    const vector<Variable>& parameters,
    float learningRate,
    float rho /* = 0.99 */,
    float epsilon /* = 1e-8 */,
    float weightDecay /* = 0 */,
    bool use_first /* = false */)
    : FirstOrderOptimizer(parameters, learningRate),
      useFirst_(use_first),
      rho_(rho),
      eps_(epsilon),
      wd_(weightDecay),
      first_(),
      second_() {
  if (useFirst_) {
    first_.reserve(parameters.size());
  }
  second_.reserve(parameters.size());

  for (const auto& parameter : parameters_) {
    if (useFirst_) {
      first_.emplace_back(af::constant(0, parameter.dims(), parameter.type()));
      fl::eval(first_.back());
    }

    second_.emplace_back(af::constant(0, parameter.dims(), parameter.type()));
    fl::eval(second_.back());
  }
}

void RMSPropOptimizer::step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    if (!parameters_[i].isGradAvailable()) {
      continue;
    }

    const af::array& grad = parameters_[i].grad().array();
    af::array& data = parameters_[i].array();

    if (wd_ != 0) {
      // Weight decay term
      data = data - wd_ * data;
    }

    af::array& second = second_[i];
    second = rho_ * second + (1 - rho_) * grad * grad;
    fl::eval(second);

    // Create shallow copy of second so that we don't update
    // "second" below
    af::array moments = second;
    if (useFirst_) {
      af::array& first = first_[i];
      first = rho_ * first + (1 - rho_) * grad;
      moments = moments - first * first;
      fl::eval(first);
    }

    data = data - (lr_ * grad) / (af::sqrt(moments) + eps_);

    fl::eval(data);
  }
}

std::string RMSPropOptimizer::prettyString() const {
  std::ostringstream ss;
  ss << "RMSProp";

  if (wd_ != 0) {
    ss << " (weight decay=" << wd_ << ")";
  }

  if (useFirst_) {
    ss << " (use first moment)";
  }

  return ss.str();
}

} // namespace fl
