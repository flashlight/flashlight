/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Dropout.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"

namespace fl {

Dropout::Dropout(double drop_ratio) : ratio_(drop_ratio) {}

Variable Dropout::forward(const Variable& input) {
  if (train_) {
    return dropout(input, ratio_);
  } else {
    return input;
  }
}

std::string Dropout::prettyString() const {
  return ("Dropout (" + std::to_string(ratio_) + ")");
}

} // namespace fl
