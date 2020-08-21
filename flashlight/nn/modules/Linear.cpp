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

#include <stdexcept>

#include "flashlight/nn/modules/Linear.h"

#include "flashlight/autograd/Functions.h"
#include "flashlight/nn/Init.h"
#include "flashlight/nn/Utils.h"

namespace fl {

Linear::Linear(int input_size, int output_size, bool bias)
    : UnaryModule(), nIn_(input_size), nOut_(output_size), bias_(bias) {
  initialize();
}

Linear::Linear(const Variable& w)
    : UnaryModule({w}), nIn_(w.dims(1)), nOut_(w.dims(0)), bias_(false) {}

Linear::Linear(const Variable& w, const Variable& b)
    : UnaryModule({w, b}), nIn_(w.dims(1)), nOut_(w.dims(0)), bias_(true) {
  if (b.dims(0) != w.dims(0)) {
    throw std::invalid_argument(
        "dimension mismatch between Linear weight and bias");
  }
}

Variable Linear::forward(const Variable& input) {
  if (bias_) {
    return linear(input, params_[0], params_[1]);
  }
  return linear(input, params_[0]);
}

void Linear::initialize() {
  int fanIn = nIn_;
  auto w = Variable(
      2 * af::kaimingNormal(af::dim4(nOut_, nIn_), fanIn, af::dtype::f32),
      true);
  if (bias_) {
    double bound = std::sqrt(1.0 / fanIn);
    auto b = uniform(af::dim4(nOut_), -bound, bound, af::dtype::f32, true);
    params_ = {w, b};
  } else {
    params_ = {w};
  }
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
