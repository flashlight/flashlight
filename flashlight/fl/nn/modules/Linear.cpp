/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Linear.h"

#include <cmath>
#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

Linear::Linear(int input_size, int output_size, bool bias)
    : UnaryModule(), nIn_(input_size), nOut_(output_size), bias_(bias) {
  initialize();
}

Linear::Linear(const Variable& w)
    : UnaryModule({w}), nIn_(w.dim(1)), nOut_(w.dim(0)), bias_(false) {}

Linear::Linear(const Variable& w, const Variable& b)
    : UnaryModule({w, b}), nIn_(w.dim(1)), nOut_(w.dim(0)), bias_(true) {
  if (b.dim(0) != w.dim(0)) {
    throw std::invalid_argument(
        "dimension mismatch between Linear weight and bias");
  }
}

Linear::Linear(const Linear& other)
    : UnaryModule(other.copyParams()),
      nIn_(other.nIn_),
      nOut_(other.nOut_),
      bias_(other.bias_) {
  train_ = other.train_;
}

Linear& Linear::operator=(const Linear& other) {
  params_ = other.copyParams();
  train_ = other.train_;
  nIn_ = other.nIn_;
  nOut_ = other.nOut_;
  bias_ = other.bias_;
  return *this;
}

Variable Linear::forward(const Variable& input) {
  if (bias_) {
    return linear(
        input,
        params_[0].astype(input.type()),
        params_[1].astype(input.type()));
  }
  return linear(input, params_[0].astype(input.type()));
}

void Linear::initialize() {
  int fanIn = nIn_;
  auto w = Variable(
      detail::kaimingUniform(Shape({nOut_, nIn_}), fanIn, fl::dtype::f32),
      true);
  if (bias_) {
    double bound = std::sqrt(1.0 / fanIn);
    auto b = uniform(Shape({nOut_}), -bound, bound, fl::dtype::f32, true);
    params_ = {w, b};
  } else {
    params_ = {w};
  }
}

std::unique_ptr<Module> Linear::clone() const {
  return std::make_unique<Linear>(*this);
}

std::string Linear::prettyString() const {
  std::ostringstream ss;
  ss << "Linear";
  ss << " (" << nIn_ << "->" << nOut_ << ")";
  if (bias_) {
    ss << " (with bias)";
  } else {
    ss << " (without bias)";
  }
  return ss.str();
}

} // namespace fl
