/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
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


#include "flashlight/optim/Optimizers.h"

#include <cmath>

using std::vector;

// References:
// SGD and Momentum: http://cs231n.github.io/neural-networks-3/#sgd
// Adam: https://arxiv.org/pdf/1412.6980.pdf
// RMSProp: https://arxiv.org/pdf/1308.0850v5.pdf

// Comparision between various update rules:
// https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM

namespace fl {

FirstOrderOptimizer::FirstOrderOptimizer(
    const vector<Variable>& parameters, double learning_rate)
    : parameters_(parameters.begin(), parameters.end()), lr_(learning_rate) {}

void FirstOrderOptimizer::zeroGrad() {
  for (auto& parameter : parameters_) {
    parameter.zeroGrad();
  }
}

SGDOptimizer::SGDOptimizer(
    const vector<Variable>& parameters,
    double learning_rate,
    double momentum /* = 0 */,
    double weight_decay /* = 0 */,
    bool use_nesterov /* = false */)
    : FirstOrderOptimizer(parameters, learning_rate),
      useNesterov_(use_nesterov),
      mu_(momentum),
      wd_(weight_decay),
      velocities_() {
  if (momentum != 0) {
    velocities_.reserve(parameters.size());
    for (const auto& parameter : parameters_) {
      velocities_.push_back(
          af::constant(0, parameter.dims(), parameter.type()));
      velocities_.back().eval();
    }
  }
}

void SGDOptimizer::step() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    if (!parameters_[i].isGradAvailable()) {
      continue;
    }

    af::array& grad = parameters_[i].grad().array();
    af::array& data = parameters_[i].array();

    if (wd_ != 0) {
      // Weight decay term
      grad = grad + wd_ * data;
    }

    if (mu_ != 0) {
      af::array& velocity = velocities_[i];

      // Regular momentum
      velocity = mu_ * velocity + grad;
      af::eval(velocity);
      if (useNesterov_) {
        // Update for nesterov momentum
        grad += velocity * mu_;
      } else {
        grad = velocity;
      }
    }
    data = data - lr_ * grad;
    af::eval(data);
  }
}

AdamOptimizer::AdamOptimizer(
    const vector<Variable>& parameters,
    double learning_rate,
    double beta1 /* = 0.9 */,
    double beta2 /* = 0.999 */,
    double epsilon /* = 1e-8 */,
    double weight_decay /* = 0 */)
    : FirstOrderOptimizer(parameters, learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      eps_(epsilon),
      wd_(weight_decay),
      count_(0),
      biasedFirst_(),
      biasedSecond_() {
  biasedFirst_.reserve(parameters.size());
  biasedSecond_.reserve(parameters.size());

  for (const auto& parameter : parameters_) {
    biasedFirst_.push_back(
        af::constant(0, parameter.dims(), parameter.type()));
    biasedSecond_.push_back(
        af::constant(0, parameter.dims(), parameter.type()));

    biasedFirst_.back().eval();
    biasedSecond_.back().eval();
  }
}

void AdamOptimizer::step() {
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

    af::array& biased_first = biasedFirst_[i];
    af::array& biased_second = biasedSecond_[i];

    biased_first = beta1_ * biased_first + (1 - beta1_) * grad;
    biased_second = beta2_ * biased_second + (1 - beta2_) * grad * grad;

    af::eval(biased_first);
    af::eval(biased_second);

    count_++;

    double corrected_bias1 = 1 - std::pow(beta1_, count_);
    double corrected_bias2 = 1 - std::pow(beta2_, count_);
    double corrected_lr = lr_ * std::sqrt(corrected_bias2) / corrected_bias1;

    data = data -
        (corrected_lr * biased_first) / (af::sqrt(biased_second) + eps_);

    af::eval(data);
  }
}

RMSPropOptimizer::RMSPropOptimizer(
    const vector<Variable>& parameters,
    double learning_rate,
    double rho /* = 0.99 */,
    double epsilon /* = 1e-8 */,
    double weight_decay /* = 0 */,
    bool use_first /* = false */)
    : FirstOrderOptimizer(parameters, learning_rate),
      useFirst_(use_first),
      rho_(rho),
      eps_(epsilon),
      wd_(weight_decay),
      first_(),
      second_() {
  if (useFirst_) {
    first_.reserve(parameters.size());
  }
  second_.reserve(parameters.size());

  for (const auto& parameter : parameters_) {
    if (useFirst_) {
      first_.push_back(af::constant(0, parameter.dims(), parameter.type()));
      first_.back().eval();
    }

    second_.push_back(af::constant(0, parameter.dims(), parameter.type()));
    second_.back().eval();
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
    af::eval(second);

    // Create shallow copy of second so that we don't update
    // "second" below
    af::array moments = second;
    if (useFirst_) {
      af::array& first = first_[i];
      first = rho_ * first + (1 - rho_) * grad;
      moments = moments - first * first;
      af::eval(first);
    }

    data = data - (lr_ * grad) / (af::sqrt(moments) + eps_);

    af::eval(data);
  }
}

} // namespace fl
