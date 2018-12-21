/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Utils.h"

#include <af/internal.h>

#include "Exception.h"

namespace fl {

bool allClose(
    const af::array& a,
    const af::array& b,
    double absTolerance /* = 1e-5 */) {
  if (a.type() != b.type()) {
    return false;
  }
  if (a.dims() != b.dims()) {
    return false;
  }
  if (a.isempty() && b.isempty()) {
    return true;
  }
  return af::max<double>(af::abs(a - b)) < absTolerance;
}

namespace detail {

void assertLinear(const af::array& arr) {
  AFML_ASSERT(af::isLinear(arr), "Array is not linear", AF_ERR_ARG);
}

} // namespace detail
} // namespace fl
