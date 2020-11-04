/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/optim/NAGOptimizer.h"

#include <cmath>

using std::vector;

namespace fl {

NAGOptimizer::NAGOptimizer(
    const vector<Variable>& parameters,
    float learningRate,
    float momentum /* = 0 */,
    float weightDecay /* = 0 */)
    : FirstOrderOptimizer(parameters, learningRate),
      mu_(momentum),
      wd_(weightDecay),
      velocities_(),
      oldLr_(learningRate) {
  if (momentum <= 0) {
    throw std::runtime_error(
        "Invalid momentum for NAG optimizer, it should be > 0");
  }
  velocities_.reserve(parameters.size());
  for (const auto& parameter : parameters_) {
    velocities_.emplace_back(
        af::constant(0, parameter.dims(), parameter.type()));
    velocities_.back().eval();
  }
}

void NAGOptimizer::step() {
  float correctedLr = lr_ / oldLr_;

  for (size_t i = 0; i < parameters_.size(); i++) {
    if (!parameters_[i].isGradAvailable()) {
      continue;
    }

    af::array& grad = parameters_[i].grad().array();
    af::array& data = parameters_[i].array();

    if (wd_ != 0) {
      // Weight decay term
      data = data * (1 - lr_ * wd_);
    }
    af::array& velocity = velocities_[i];
    // this velocity corresponds to fairseq velocity * -1
    velocity = mu_ * velocity * correctedLr + lr_ * grad;
    af::eval(velocity);
    grad = grad * lr_ + velocity * mu_;
    data = data - grad;
    af::eval(data);
  }
  oldLr_ = lr_;
}

std::string NAGOptimizer::prettyString() const {
  std::ostringstream ss;
  ss << "NAG (lr=" << lr_ << " ); (previous lr=" << oldLr_ << ");";

  if (wd_ != 0) {
    ss << " (weight decay=" << wd_ << ");";
  }
  ss << " (Nesterov momentum=" << mu_ << ")";
  return ss.str();
}

} // namespace fl
