/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
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
    const vector<Variable>& parameters,
    double learningRate)
    : parameters_(parameters.begin(), parameters.end()), lr_(learningRate) {}

void FirstOrderOptimizer::zeroGrad() {
  for (auto& parameter : parameters_) {
    parameter.zeroGrad();
  }
}

SGDOptimizer::SGDOptimizer(
    const vector<Variable>& parameters,
    double learningRate,
    double momentum /* = 0 */,
    double weightDecay /* = 0 */,
    bool useNesterov /* = false */)
    : FirstOrderOptimizer(parameters, learningRate),
      useNesterov_(useNesterov),
      mu_(momentum),
      wd_(weightDecay),
      velocities_() {
  if (momentum != 0) {
    velocities_.reserve(parameters.size());
    for (const auto& parameter : parameters_) {
      velocities_.emplace_back(
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

AdamOptimizer::AdamOptimizer(
    const vector<Variable>& parameters,
    double learningRate,
    double beta1 /* = 0.9 */,
    double beta2 /* = 0.999 */,
    double epsilon /* = 1e-8 */,
    double weightDecay /* = 0 */)
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

    af::array& biasedFirst = biasedFirst_[i];
    af::array& biasedSecond = biasedSecond_[i];

    biasedFirst = beta1_ * biasedFirst + (1 - beta1_) * grad;
    biasedSecond = beta2_ * biasedSecond + (1 - beta2_) * grad * grad;

    af::eval(biasedFirst);
    af::eval(biasedSecond);

    count_++;

    double correctedBias1 = 1 - std::pow(beta1_, count_);
    double correctedBias2 = 1 - std::pow(beta2_, count_);
    double correctedLr = lr_ * std::sqrt(correctedBias2) / correctedBias1;

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

RMSPropOptimizer::RMSPropOptimizer(
    const vector<Variable>& parameters,
    double learningRate,
    double rho /* = 0.99 */,
    double epsilon /* = 1e-8 */,
    double weightDecay /* = 0 */,
    bool use_first /* = false */)
    : FirstOrderOptimizer(parameters, learningRate),
      useFirst_(use_first),
      rho_(rho),
      eps_(epsilon),
      wd_(weightDecay),
      first_(),
      second_() {
  if (useFirst_) {
    first_.reserve(parameters.size());
  }
  second_.reserve(parameters.size());

  for (const auto& parameter : parameters_) {
    if (useFirst_) {
      first_.emplace_back(af::constant(0, parameter.dims(), parameter.type()));
      first_.back().eval();
    }

    second_.emplace_back(af::constant(0, parameter.dims(), parameter.type()));
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

std::string RMSPropOptimizer::prettyString() const {
  std::ostringstream ss;
  ss << "RMSProp";

  if (wd_ != 0) {
    ss << " (weight decay=" << wd_ << ")";
  }

  if (useFirst_) {
    ss << " (use first moment)";
  }

  return ss.str();
}

AdadeltaOptimizer::AdadeltaOptimizer(
    const std::vector<Variable>& parameters,
    double learningRate /* = 1.0 */,
    double rho /* = 0.9 */,
    double epsilon /* = 1e-8 */,
    double weightDecay /* = 0 */)
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

    accGrad_.back().eval();
    accDelta_.back().eval();
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
    af::eval(accGrad);

    auto delta = af::sqrt(accDelta + eps_) / af::sqrt(accGrad + eps_) * grad;

    data = data - lr_ * delta;
    af::eval(data);

    accDelta = rho_ * accDelta + (1 - rho_) * delta * delta;
    af::eval(accDelta);
  }
}

std::string AdadeltaOptimizer::prettyString() const {
  std::ostringstream ss;
  ss << "Adadelta";

  if (wd_ != 0) {
    ss << " (weight decay=" << wd_ << ")";
  }

  return ss.str();
}

} // namespace fl
