/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Reorder.h"

#include <stdexcept>
#include <utility>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"

namespace fl {

Reorder::Reorder(Shape shape) : shape_(std::move(shape)) {}

Variable Reorder::forward(const Variable& input) {
  if (input.ndim() != shape_.ndim()) {
    throw std::invalid_argument(
        "Reorder::forward - input tensor has different "
        "number of dimensions than reorder shape.");
  }
  return reorder(input, shape_);
}

std::string Reorder::prettyString() const {
  std::ostringstream ss;
  ss << "Reorder";
  ss << " (" << shape_ << ")";
  return ss.str();
}

} // namespace fl
