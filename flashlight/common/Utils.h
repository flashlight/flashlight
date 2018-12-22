/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>

namespace fl {

/**
 * Returns true if two arrays are of same type and are element-wise equal within
 * given tolerance limit.
 *
 * @param [a,b] input arrays to compare
 * @param absTolerance absolute tolerance allowed
 *
 */
bool allClose(
    const af::array& a,
    const af::array& b,
    double absTolerance = 1e-5);

} // namespace fl
