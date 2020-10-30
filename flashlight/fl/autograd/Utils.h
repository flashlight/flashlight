/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/autograd/Variable.h"

namespace fl {

/**
 * \defgroup autograd_utils Autograd Utils
 * @{
 */

/**
 * Returns true if two Variable are of same type and are element-wise equal
 * within given tolerance limit.
 *
 * @param [a,b] input Variables to compare
 * @param absTolerance absolute tolerance allowed
 *
 */
bool allClose(const Variable& a, const Variable& b, double absTolerance = 1e-5);

/** @} */

} // namespace fl
