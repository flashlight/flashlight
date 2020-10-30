/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "flashlight/fl/optim/NovogradOptimizer.h"

#include <cmath>

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
    accGrad_.emplace_back(af::constant(0, parameter.dims(), parameter.type()));

    accGrad_.back().eval();
  }
}

void NovogradOptimizer::step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    if (!parameters_[i].isGradAvailable()) {
      continue;
    }

    const af::array& grad = parameters_[i].grad().array();
    af::array& data = parameters_[i].array();
    af::array& accGrad = accGrad_[i];

    double gradNorm = af::sum<double>(grad * grad);

    accGradNorm_[i] = beta2_ * accGradNorm_[i] + (1 - beta2_) * gradNorm;
    accGrad = beta1_ * accGrad +
        (1 - beta1_) *
            (grad / (static_cast<float>(std::sqrt(accGradNorm_[i]) + eps_)) +
             wd_ * data);
    af::eval(accGrad);

    data = data - (lr_ * accGrad);

    af::eval(data);
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
