/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
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
    accGrad_.emplace_back(af::constant(0, parameter.dims(), parameter.type()));
    accDelta_.emplace_back(af::constant(0, parameter.dims(), parameter.type()));

    fl::eval(accGrad_.back());
    fl::eval(accDelta_.back());
  }
}

void AdadeltaOptimizer::step() {
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

    af::array& accGrad = accGrad_[i];
    af::array& accDelta = accDelta_[i];

    accGrad = rho_ * accGrad + (1 - rho_) * grad * grad;
    fl::eval(accGrad);

    auto delta = af::sqrt(accDelta + eps_) / af::sqrt(accGrad + eps_) * grad;

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
