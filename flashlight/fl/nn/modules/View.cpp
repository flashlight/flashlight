/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/View.h"

#include <utility>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

View::View(Shape dims) : dims_(std::move(dims)) {}

Variable View::forward(const Variable& input) {
  Shape dims = dims_;
  return moddims(input, dims);
}

std::unique_ptr<Module> View::clone() const {
  return std::make_unique<View>(*this);
}

std::string View::prettyString() const {
  std::ostringstream ss;
  ss << "View (" << dims_ << ")";
  return ss.str();
}

} // namespace fl
