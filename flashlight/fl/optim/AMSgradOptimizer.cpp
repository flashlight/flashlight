/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/optim/AMSgradOptimizer.h"

#include <cmath>

#include "flashlight/fl/tensor/Compute.h"

using std::vector;

namespace fl {

AMSgradOptimizer::AMSgradOptimizer(
    const vector<Variable>& parameters,
    float learningRate,
    float beta1 /* = 0.9 */,
    float beta2 /* = 0.999 */,
    float epsilon /* = 1e-8 */,
    float weightDecay /* = 0 */)
    : FirstOrderOptimizer(parameters, learningRate),
      beta1_(beta1),
      beta2_(beta2),
      eps_(epsilon),
      wd_(weightDecay),
      biasedFirst_(),
      biasedSecond_(),
      maxExpAvgSq_() {
  biasedFirst_.reserve(parameters.size());
  biasedSecond_.reserve(parameters.size());
  maxExpAvgSq_.reserve(parameters.size());

  for (const auto& parameter : parameters_) {
    biasedFirst_.emplace_back(fl::full(parameter.shape(), 0, parameter.type()));
    biasedSecond_.emplace_back(fl::full(parameter.shape(), 0, parameter.type()));
    maxExpAvgSq_.emplace_back(fl::full(parameter.shape(), 0, parameter.type()));

    fl::eval(biasedFirst_.back());
    fl::eval(biasedSecond_.back());
    fl::eval(maxExpAvgSq_.back());
  }
}

void AMSgradOptimizer::step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    if (!parameters_[i].isGradAvailable()) {
      continue;
    }

    const Tensor& grad = parameters_[i].grad().tensor();
    Tensor& data = parameters_[i].tensor();

    if (wd_ != 0) {
      data = data - wd_ * data;
    }

    Tensor& biasedFirst = biasedFirst_[i];
    Tensor& biasedSecond = biasedSecond_[i];
    Tensor& maxExpAvgSq = maxExpAvgSq_[i];

    biasedFirst = beta1_ * biasedFirst + (1 - beta1_) * grad;
    biasedSecond = beta2_ * biasedSecond + (1 - beta2_) * grad * grad;
    maxExpAvgSq = fl::maximum(maxExpAvgSq, biasedSecond);
    fl::eval(biasedFirst);
    fl::eval(biasedSecond);
    fl::eval(maxExpAvgSq);

    data = data - (lr_ * biasedFirst) / (fl::sqrt(maxExpAvgSq) + eps_);

    fl::eval(data);
  }
}

std::string AMSgradOptimizer::prettyString() const {
  std::ostringstream ss;
  ss << "AMSgrad from ";

  if (wd_ != 0) {
    ss << " (weight decay=" << wd_ << ")";
  }

  return ss.str();
}

} // namespace fl
