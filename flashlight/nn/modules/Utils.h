/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "Module.h"

#include <flashlight/common/Defines.h>

namespace fl {

/**
 * Returns true if the parameters of two modules are of same type and are
 * element-wise equal within given tolerance limit.
 *
 * @param [a,b] input Modules to compare
 * @param absTolerance absolute tolerance allowed
 *
 */
bool allParamsClose(
    const Module& a,
    const Module& b,
    double absTolerance = 1e-5);

namespace detail {
struct IntOrPadMode {
  /* implicit */ IntOrPadMode(int val) : padVal(val) {}
  /* implicit */ IntOrPadMode(PaddingMode mode)
      : padVal(static_cast<int>(mode)) {}
  const int padVal;
};
} // namespace detail

} // namespace fl
