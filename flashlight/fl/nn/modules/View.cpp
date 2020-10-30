/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/View.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

View::View(af::dim4 dims) : dims_(dims) {}

Variable View::forward(const Variable& input) {
  af::dim4 dims = dims_;
  return moddims(input, dims);
}

std::string View::prettyString() const {
  std::ostringstream ss;
  ss << "View (" << dims_ << ")";
  return ss.str();
}

} // namespace fl
