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

#include "Dropout.h"

#include <flashlight/autograd/Functions.h>
#include <flashlight/nn/Init.h>

namespace fl {

Dropout::Dropout(double drop_ratio) : ratio_(drop_ratio) {}

Variable Dropout::forward(const Variable& input) {
  if (train_) {
    return (uniform(input.dims(), 0.0, 1.0, f32, false) > ratio_) *
        (1.0 / (1.0 - ratio_)) * input;
  } else {
    return input;
  }
}

std::string Dropout::prettyString() const {
  return ("Dropout (" + std::to_string(ratio_) + ")");
}

} // namespace fl
