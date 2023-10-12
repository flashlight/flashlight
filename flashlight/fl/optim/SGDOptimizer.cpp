/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/optim/SGDOptimizer.h"

#include <cmath>

#include "flashlight/fl/tensor/Compute.h"

using std::vector;

namespace fl {

SGDOptimizer::SGDOptimizer(
    const vector<Variable>& parameters,
    float learningRate,
    float momentum /* = 0 */,
    float weightDecay /* = 0 */,
    bool useNesterov /* = false */)
    : FirstOrderOptimizer(parameters, learningRate),
      useNesterov_(useNesterov),
      mu_(momentum),
      wd_(weightDecay),
      velocities_() {
  if (momentum != 0) {
    velocities_.reserve(parameters.size());
    for (const auto& parameter : parameters_) {
      velocities_.emplace_back(fl::full(parameter.shape(), 0, parameter.type()));
      fl::eval(velocities_.back());
    }
  }
}

void SGDOptimizer::step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    if (!parameters_[i].isGradAvailable()) {
      continue;
    }

    Tensor& grad = parameters_[i].grad().tensor();
    Tensor& data = parameters_[i].tensor();

    if (wd_ != 0) {
      // Weight decay term
      grad = grad + wd_ * data;
    }

    if (mu_ != 0) {
      Tensor& velocity = velocities_[i];

      // Regular momentum
      velocity = mu_ * velocity + grad;
      fl::eval(velocity);
      if (useNesterov_) {
        // Update for nesterov momentum
        grad += velocity * mu_;
      } else {
        grad = velocity;
      }
    }
    data = data - lr_ * grad;
    fl::eval(data);
  }
}

std::string SGDOptimizer::prettyString() const {
  std::ostringstream ss;
  ss << "SGD";

  if (wd_ != 0) {
    ss << " (weight decay=" << wd_ << ")";
  }
  if (useNesterov_ && mu_ != 0) {
    ss << " (Nesterov momentum=" << mu_ << ")";
  } else if (mu_ != 0) {
    ss << " (momentum=" << mu_ << ")";
  }

  return ss.str();
}

} // namespace fl
