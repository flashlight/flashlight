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

#include "flashlight/fl/optim/AdamOptimizer.h"

#include <cmath>

using std::vector;

namespace fl {

AdamOptimizer::AdamOptimizer(
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
      count_(0),
      biasedFirst_(),
      biasedSecond_() {
  biasedFirst_.reserve(parameters.size());
  biasedSecond_.reserve(parameters.size());

  for (const auto& parameter : parameters_) {
    biasedFirst_.emplace_back(
        af::constant(0, parameter.dims(), parameter.type()));
    biasedSecond_.emplace_back(
        af::constant(0, parameter.dims(), parameter.type()));

    biasedFirst_.back().eval();
    biasedSecond_.back().eval();
  }
}

void AdamOptimizer::step() {
  count_++;
  float correctedBias1 = 1 - std::pow(beta1_, count_);
  float correctedBias2 = 1 - std::pow(beta2_, count_);
  float correctedLr = lr_ * std::sqrt(correctedBias2) / correctedBias1;

  for (size_t i = 0; i < parameters_.size(); i++) {
    if (!parameters_[i].isGradAvailable()) {
      continue;
    }

    const af::array& grad = parameters_[i].grad().array();
    af::array& data = parameters_[i].array();

    if (wd_ != 0) {
      // Weight decay term
      data = data - wd_ * lr_ * data;
    }

    af::array& biasedFirst = biasedFirst_[i];
    af::array& biasedSecond = biasedSecond_[i];

    biasedFirst = beta1_ * biasedFirst + (1 - beta1_) * grad;
    biasedSecond = beta2_ * biasedSecond + (1 - beta2_) * grad * grad;

    af::eval(biasedFirst);
    af::eval(biasedSecond);

    data = data - (correctedLr * biasedFirst) / (af::sqrt(biasedSecond) + eps_);

    af::eval(data);
  }
}

std::string AdamOptimizer::prettyString() const {
  std::ostringstream ss;
  ss << "Adam";

  if (wd_ != 0) {
    ss << " (weight decay=" << wd_ << ")";
  }

  return ss.str();
}

} // namespace fl
