/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/Utils.h"
#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

bool allClose(
    const Variable& a,
    const Variable& b,
    double absTolerance /* = 1e-5 */) {
  return allClose(a.tensor(), b.tensor(), absTolerance);
}

} // namespace fl
