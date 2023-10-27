/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/Utils.h"

#include <vector>

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

namespace detail {

std::ostream& operator<<(std::ostream& os, const UseVal& useVal) {
  os << "(" << useVal.user << ", " << useVal.inputIdx << ")";
  return os;
}

} // namespace detail

bool operator==(const UseList& actual, const UseValList& expect) {
  if (actual.size() != expect.size()) {
    return false;
  }
  unsigned i = 0;
  for (const auto& actualUse : actual) {
    const auto& expectUse = expect[i];
    if (&actualUse->user() != expectUse.user.get() ||
        actualUse->inputIdx() != expectUse.inputIdx) {
      return false;
    }
    i++;
  }
  return true;
}

std::ostream& operator<<(std::ostream& os, const Use* use) {
  os << "(" << &use->user() << ", " << use->inputIdx() << ")";
  return os;
}

} // namespace fl
