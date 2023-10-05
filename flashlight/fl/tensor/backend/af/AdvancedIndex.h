/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <vector>

namespace af {
class array;
class dim4;
} // namespace af

namespace fl {

namespace detail {

/**
 * This function computes the gradient of an indexing operator
 * when the af::index variable is an array (advanced indexing)
 * @param inp The input af::array which is the gradient of output of index
 * operator
 * @param idxStart The starting index of each dimension of index operator
 * @param idxEnd The ending index of each dimension of index operator
 * @param outDims The dimensions of output af::array which is the input of index
 * oeprator
 * @param idxArr The pointer to the advanced index of every dimension of index
 * operator
 * @param out The output Varible which is the gradient of input of index
 * operator
 */
void advancedIndex(
    const af::array& inp,
    const af::dim4& idxStart,
    const af::dim4& idxEnd,
    const af::dim4& outDims,
    const std::vector<af::array>& idxArr,
    af::array& out);

} // namespace detail
} // namespace fl
