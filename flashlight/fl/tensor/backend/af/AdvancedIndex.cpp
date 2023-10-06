/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <af/array.h>

#include <stdexcept>
#include <vector>

namespace fl {
namespace detail {

void advancedIndex(
    const af::array& inp,
    const af::dim4& idxStart,
    const af::dim4& idxEnd,
    const af::dim4& outDims,
    const std::vector<af::array>& idxArr,
    af::array& out) {
  throw std::runtime_error("gradAdvancedIndex not implemented for cpu");
}

} // namespace detail
} // namespace fl
