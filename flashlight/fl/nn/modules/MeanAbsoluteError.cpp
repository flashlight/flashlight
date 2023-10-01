/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/MeanAbsoluteError.h"

#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"

namespace fl {

Variable MeanAbsoluteError::forward(
    const Variable& inputs,
    const Variable& targets) {
  if (inputs.shape() != targets.shape()) {
    throw std::invalid_argument(
        "MeanAbsoluteError::forward - inputs and targets are of different"
        " sizes: {inputs: " +
        inputs.shape().toString() + ", targets: " + targets.shape().toString() +
        "}");
  }

  auto df = inputs - targets;
  return mean(flat(fl::abs(df)), {0});
}

std::string MeanAbsoluteError::prettyString() const {
  return "MeanAbsoluteError";
}

} // namespace fl
