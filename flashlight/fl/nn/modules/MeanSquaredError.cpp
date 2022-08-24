/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/MeanSquaredError.h"

#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"

namespace fl {

Variable MeanSquaredError::forward(
    const Variable& inputs,
    const Variable& targets) {
  if (inputs.shape() != targets.shape()) {
    throw std::invalid_argument(
        "MeanSquaredError::forward - inputs and targets are of different"
        " sizes: {inputs: " +
        inputs.shape().toString() + ", targets: " + targets.shape().toString() +
        "}");
  }

  auto df = inputs - targets;
  auto res = mean(flat(df * df), {0});
  return res;
}

std::string MeanSquaredError::prettyString() const {
  return "MeanSquaredError";
}

} // namespace fl
