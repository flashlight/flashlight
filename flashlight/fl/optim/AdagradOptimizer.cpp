/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/optim/AdagradOptimizer.h"

#include <cmath>

#include "flashlight/fl/tensor/Compute.h"

namespace fl {

AdagradOptimizer::AdagradOptimizer(
    const std::vector<Variable>& parameters,
    float learningRate /* = 1.0 */,
    float epsilon /* = 1e-8 */,
    float weightDecay /* = 0 */)
    : FirstOrderOptimizer(parameters, learningRate),
      eps_(epsilon),
      wd_(weightDecay) {
  variance_.reserve(parameters.size());
  for (const auto& param : parameters_) {
    variance_.push_back(fl::full(param.shape(), 0, param.type()));
    fl::eval(variance_.back());
  }
}

void AdagradOptimizer::step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    if (!parameters_[i].isGradAvailable()) {
      continue;
    }

    const Tensor& grad = parameters_[i].grad().tensor();
    Tensor& data = parameters_[i].tensor();
    Tensor& variance = variance_[i];

    if (wd_ != 0) {
      // Weight decay term
      data = data - wd_ * data;
    }

    variance = variance + grad * grad;
    fl::eval(variance);
    data = data - lr_ * grad / (fl::sqrt(variance) + eps_);
    fl::eval(data);
  }
}

std::string AdagradOptimizer::prettyString() const {
  std::ostringstream ss;
  ss << "Adagrad";

  if (eps_ != 0) {
    ss << " (epsilon=" << eps_ << ")";
  }

  return ss.str();
}

} // namespace fl
