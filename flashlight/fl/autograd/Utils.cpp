/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/Utils.h"

#include "flashlight/fl/common/Utils.h"

namespace fl {

bool allClose(
    const Variable& a,
    const Variable& b,
    double absTolerance /* = 1e-5 */) {
  return allClose(a.array(), b.array(), absTolerance);
}

} // namespace fl
