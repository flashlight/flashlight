/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <af/blas.h>

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

namespace fl {

Tensor ArrayFireBackend::matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp) {
  unsigned numDims = std::max(lhs.ndim(), rhs.ndim());
  if ((lhs.ndim() == 1 || rhs.ndim() == 1) && numDims > 1) {
    numDims -= 1;
  }

  af::array lhsArray = toArray(lhs);
  af::array rhsArray = toArray(rhs);

  if (lhs.ndim() == 1 && rhs.ndim() == 1) {
    // Simulate a dot product by transpoing the lhs:
    // (1, k) x (k, 1) --> (1, 1) --> reshape to (1)
    // Ignore other transposes since 1D tensors are the transpose of themselves.
    // ArrayFire would otherwise transpose a (k) tensor to (1, k) since (k) =
    // (k, 1, 1, 1) and ArrayFire transpose transposes the first two dimensions.
    lhsProp = MatrixProperty::Transpose;
    rhsProp = MatrixProperty::None;
    numDims = 1;
  } else {
    if (rhs.ndim() == 1) {
      rhsArray = af::moddims(toArray(rhs), {rhs.dim(0), 1});
    }
    if (lhs.ndim() == 1) {
      lhsArray = af::moddims(toArray(lhs), {1, lhs.dim(0)});
    }
  }

  auto arr = af::matmul(
      lhsArray,
      rhsArray,
      detail::flToAfMatrixProperty(lhsProp),
      detail::flToAfMatrixProperty(rhsProp));

  if (lhs.ndim() == 1 && rhs.ndim() == 2) {
    arr = af::moddims(arr, arr.dims(1));
  }

  return toTensor<ArrayFireTensor>(std::move(arr), numDims);
}
} // namespace fl
