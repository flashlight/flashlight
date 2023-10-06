/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/optim/NovogradOptimizer.h"

#include <cmath>

#include "flashlight/fl/tensor/Compute.h"

using std::vector;

namespace fl {

NovogradOptimizer::NovogradOptimizer(
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
      accGradNorm_(),
      accGrad_() {
  accGradNorm_.reserve(1);
  accGrad_.reserve(parameters.size());

  for (const auto& parameter : parameters_) {
    accGradNorm_.emplace_back(0.0);
    accGrad_.emplace_back(fl::full(parameter.shape(), 0, parameter.type()));

    fl::eval(accGrad_.back());
  }
}

void NovogradOptimizer::step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    if (!parameters_[i].isGradAvailable()) {
      continue;
    }

    const Tensor& grad = parameters_[i].grad().tensor();
    Tensor& data = parameters_[i].tensor();
    Tensor& accGrad = accGrad_[i];

    double gradNorm = fl::sum(grad * grad).asScalar<double>();

    accGradNorm_[i] = beta2_ * accGradNorm_[i] + (1 - beta2_) * gradNorm;
    accGrad = beta1_ * accGrad +
        (1 - beta1_) *
            (grad / (static_cast<float>(std::sqrt(accGradNorm_[i]) + eps_)) +
             wd_ * data);
    fl::eval(accGrad);

    data = data - (lr_ * accGrad);

    fl::eval(data);
  }
}

std::string NovogradOptimizer::prettyString() const {
  std::ostringstream ss;
  ss << "Novograd";

  if (wd_ != 0) {
    ss << " (weight decay=" << wd_ << ")";
  }

  return ss.str();
}

} // namespace fl
