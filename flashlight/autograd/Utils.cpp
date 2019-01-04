/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/autograd/Utils.h"

#include "flashlight/common/Utils.h"

namespace fl {

bool allClose(
    const Variable& a,
    const Variable& b,
    double absTolerance /* = 1e-5 */) {
  return allClose(a.array(), b.array(), absTolerance);
}

} // namespace fl
