/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Utils.h"

#include <flashlight/autograd/Utils.h>

namespace fl {

bool allParamsClose(
    const Module& a,
    const Module& b,
    double absTolerance /* = 1e-5 */) {
  if (a.params().size() != b.params().size()) {
    return false;
  }
  const auto aParams = a.params();
  const auto bParams = b.params();
  for (int p = 0; p < aParams.size(); ++p) {
    if (!allClose(aParams[p], bParams[p], absTolerance)) {
      return false;
    }
  }
  return true;
}
} // namespace fl
