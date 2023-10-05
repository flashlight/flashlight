/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/optim/AdadeltaOptimizer.h"

#include <cmath>

#include "flashlight/fl/tensor/Compute.h"

namespace fl {

AdadeltaOptimizer::AdadeltaOptimizer(
    const std::vector<Variable>& parameters,
    float learningRate /* = 1.0 */,
    float rho /* = 0.9 */,
    float epsilon /* = 1e-8 */,
    float weightDecay /* = 0 */)
    : FirstOrderOptimizer(parameters, learningRate),
      rho_(rho),
      eps_(epsilon),
      wd_(weightDecay),
      accGrad_(),
      accDelta_() {
  accGrad_.reserve(parameters.size());
  accDelta_.reserve(parameters.size());

  for (const auto& parameter : parameters_) {
    accGrad_.emplace_back(fl::full(parameter.shape(), 0, parameter.type()));
    accDelta_.emplace_back(fl::full(parameter.shape(), 0, parameter.type()));

    fl::eval(accGrad_.back());
    fl::eval(accDelta_.back());
  }
}

void AdadeltaOptimizer::step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    if (!parameters_[i].isGradAvailable()) {
      continue;
    }

    const Tensor& grad = parameters_[i].grad().tensor();
    Tensor& data = parameters_[i].tensor();

    if (wd_ != 0) {
      // Weight decay term
      data = data - wd_ * data;
    }

    Tensor& accGrad = accGrad_[i];
    Tensor& accDelta = accDelta_[i];

    accGrad = rho_ * accGrad + (1 - rho_) * grad * grad;
    fl::eval(accGrad);

    auto delta = fl::sqrt(accDelta + eps_) / fl::sqrt(accGrad + eps_) * grad;

    data = data - lr_ * delta;
    fl::eval(data);

    accDelta = rho_ * accDelta + (1 - rho_) * delta * delta;
    fl::eval(accDelta);
  }
}

std::string AdadeltaOptimizer::prettyString() const {
  std::ostringstream ss;
  ss << "Adadelta";

  if (wd_ != 0) {
    ss << " (weight decay=" << wd_ << ")";
  }
  ss << " (rho=" << rho_ << ")";
  if (eps_ != 0) {
    ss << " (epsilon=" << eps_ << ")";
  }

  return ss.str();
}

} // namespace fl
