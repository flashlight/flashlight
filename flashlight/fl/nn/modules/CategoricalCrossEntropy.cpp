/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/CategoricalCrossEntropy.h"

#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"

namespace fl {

Variable CategoricalCrossEntropy::forward(
    const Variable& inputs,
    const Variable& targets) {
  return categoricalCrossEntropy(inputs, targets, reduction_, ignoreIndex_);
}

std::string CategoricalCrossEntropy::prettyString() const {
  return "CategoricalCrossEntropy";
}

} // namespace fl
