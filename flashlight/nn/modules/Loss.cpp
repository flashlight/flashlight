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

#include <stdexcept>

#include "flashlight/nn/modules/Loss.h"

#include "flashlight/autograd/Functions.h"

namespace fl {

Variable MeanSquaredError::forward(
    const Variable& inputs,
    const Variable& targets) {
  auto df = inputs - targets;
  auto res = mean(flat(df * df), {0});
  return res;
}

std::string MeanSquaredError::prettyString() const {
  return "MeanSquaredError";
}

Variable MeanAbsoluteError::forward(
    const Variable& inputs,
    const Variable& targets) {
  auto df = inputs - targets;
  return mean(flat(fl::abs(df)), {0});
}

std::string MeanAbsoluteError::prettyString() const {
  return "MeanAbsoluteError";
}

Variable BinaryCrossEntropy::forward(
    const Variable& inputs,
    const Variable& targets) {
  return mean(flat(binaryCrossEntropy(inputs, targets)), {0});
}

Variable BinaryCrossEntropy::forward(
    const Variable& inputs,
    const Variable& targets,
    const Variable& weights) {
  return mean(flat(weights * binaryCrossEntropy(inputs, targets)), {0});
}

std::string BinaryCrossEntropy::prettyString() const {
  return "BinaryCrossEntropy";
}

Variable CategoricalCrossEntropy::forward(
    const Variable& inputs,
    const Variable& targets) {
  return categoricalCrossEntropy(inputs, targets, reduction_);
}

std::string CategoricalCrossEntropy::prettyString() const {
  return "CategoricalCrossEntropy";
}

} // namespace fl
