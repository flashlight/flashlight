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

#include "Linear.h"

#include <flashlight/autograd/Functions.h>
#include <flashlight/common/Exception.h>
#include <flashlight/nn/Utils.h>
#include <flashlight/nn/Init.h>

namespace fl {

Linear::Linear(int input_size, int output_size, bool bias)
    : Module(), nIn_(input_size), nOut_(output_size), bias_(bias) {
  initialize();
}

Linear::Linear(const Variable& w)
    : Module({w}), nIn_(w.dims(1)), nOut_(w.dims(0)), bias_(false) {}

Linear::Linear(const Variable& w, const Variable& b)
    : Module({w, b}), nIn_(w.dims(1)), nOut_(w.dims(0)), bias_(true) {
  AFML_ASSERT(
      b.dims(0) == w.dims(0),
      "nn:Linear: Dimension mismatch between weight and bias.",
      AF_ERR_ARG);
}

Variable Linear::forward(const Variable& input) {
  if (bias_) {
    return linear(input, params_[0], params_[1]);
  }
  return linear(input, params_[0]);
}

void Linear::initialize() {
  auto w = kaimingUniform(nOut_, nIn_);
  if (bias_) {
    int fan_in = detail::computeFans(w.dims()).first;
    double bound = std::sqrt(1.0 / fan_in);
    auto b = uniform(af::dim4(nOut_), -bound, bound);
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
