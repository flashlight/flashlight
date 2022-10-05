/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/Utils.h"

#include <vector>

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

bool operator==(UseList actual, UseValList expect) {
  if (actual.size() != expect.size()) {
    return false;
  }
  unsigned i = 0;
  for (const auto& actualUse : actual) {
    const auto& expectUse = expect[i];
    if (actualUse->user() != expectUse.user ||
        actualUse->inputIdx() != expectUse.inputIdx) {
      return false;
    }
    i++;
  }
  return true;
}

} // namespace fl
