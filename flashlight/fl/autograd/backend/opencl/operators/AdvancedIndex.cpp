/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <af/array.h>

#include <stdexcept>
#include <vector>

namespace fl {

class Variable;

void gradAdvancedIndex(
    const Variable& inp,
    const af::dim4& idxStart,
    const af::dim4& idxEnd,
    const af::dim4& outDims,
    const std::vector<af::array>& idxArr,
    Variable& out) {
  throw std::runtime_error("gradAdvancedIndex not implemented for cpu");
}

} // namespace fl
