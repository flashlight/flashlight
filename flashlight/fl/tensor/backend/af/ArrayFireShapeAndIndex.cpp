/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <af/blas.h>

#include <numeric>

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

namespace fl {

Tensor ArrayFireBackend::reshape(const Tensor& tensor, const Shape& shape) {
  return toTensor<ArrayFireTensor>(
      af::moddims(toArray(tensor), detail::flToAfDims(shape)), shape.ndim());
}

Tensor ArrayFireBackend::transpose(
    const Tensor& tensor,
    const Shape& axes /* = {} */) {
  if (tensor.ndim() == 1) {
    return tensor;
  } else if (
      tensor.ndim() == 2 && (axes.ndim() == 0 || axes == Shape({1, 0}))) {
    // fastpath for matrices
    return toTensor<ArrayFireTensor>(
        af::transpose(toArray(tensor)), tensor.ndim());
  } else if (axes.ndim() == 0) {
    std::vector<Dim> dims(AF_MAX_DIMS);
    std::iota(std::begin(dims), std::end(dims), 0);
    // Compute the reversed dimensions for as many ndims as are in the input
    for (unsigned i = 0; i < tensor.ndim(); ++i) {
      dims[i] = tensor.ndim() - 1 - i;
    }

    // flip all dimensions
    return toTensor<ArrayFireTensor>(
        af::reorder(toArray(tensor), dims[0], dims[1], dims[2], dims[3]),
        tensor.ndim());
  } else {
    if (axes.ndim() > AF_MAX_DIMS) {
      throw std::invalid_argument(
          "ArrayFire tensor transpose was given "
          "permutation dims with > 4 axes");
    }
    if (axes.ndim() != tensor.ndim()) {
      throw std::invalid_argument(
          "ArrayFire tensor transpose axes don't match tensor's for "
          "permutation - axes must have the same number of "
          "dimensions as the tensor");
    }
    // reorder based on specified dimensions
    std::vector<dim_t> d(AF_MAX_DIMS);
    std::iota(std::begin(d), std::end(d), 0);
    for (size_t i = 0; i < axes.ndim(); ++i) {
      if (axes[i] > tensor.ndim() - 1) {
        throw std::invalid_argument(
            "ArrayFireBackend::transpose - given dimension is larger "
            "than the number of dimensions in the tensor");
      }

      d[i] = axes[i];
    }
    return toTensor<ArrayFireTensor>(
        af::reorder(toArray(tensor), d[0], d[1], d[2], d[3]), tensor.ndim());
  }
}

Tensor ArrayFireBackend::tile(const Tensor& tensor, const Shape& shape) {
  return toTensor<ArrayFireTensor>(
      af::tile(toArray(tensor), detail::flToAfDims(shape)),
      // TODO: check
      std::max(tensor.ndim(), shape.ndim()));
}

Tensor ArrayFireBackend::concatenate(
    const std::vector<Tensor>& tensors,
    const unsigned axis) {
  af::array out;
  switch (tensors.size()) {
    case 0:
      return toTensor<ArrayFireTensor>(ArrayFireTensor()); // empty tensor
    case 1:
      return tensors.front();
    case 2:
      out = af::join(axis, toArray(tensors[0]), toArray(tensors[1]));
      break;
    case 3:
      out = af::join(
          axis, toArray(tensors[0]), toArray(tensors[1]), toArray(tensors[2]));
      break;
    case 4:
      out = af::join(
          axis,
          toArray(tensors[0]),
          toArray(tensors[1]),
          toArray(tensors[2]),
          toArray(tensors[3]));
      break;
    default:
      // TODO: iteratively concat to remove this limitation
      throw std::invalid_argument(
          "ArrayFire concatenate doesn't support > 4 tensors");
  }

  unsigned numDims = tensors[0].ndim();
  if (axis > std::max(numDims - 1, 0u)) {
    numDims = axis + 1;
  }

  // All tensors have the same numdims else AF would throw
  return toTensor<ArrayFireTensor>(std::move(out), numDims);
}

Tensor ArrayFireBackend::nonzero(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(
      af::where(toArray(tensor)), /* numDims = */ 1);
}

Tensor ArrayFireBackend::pad(
    const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type) {
  if (padWidths.size() > AF_MAX_DIMS) {
    throw std::invalid_argument(
        "ArrayFireBackend::pad - given padWidths for more than 4 dimensions");
  }

  // convert ((begin_1, end_1), ..., (begin_k, end_k)) to ((begin_1, ...,
  // begin_k), (end_1, ..., end_k)) for ArrayFire
  af::dim4 beginPadding, endPadding;
  for (size_t i = 0; i < padWidths.size(); ++i) {
    auto& [first, second] = padWidths[i];
    beginPadding[i] = first;
    endPadding[i] = second;
  }

  return toTensor<ArrayFireTensor>(
      af::pad(
          toArray(input),
          beginPadding,
          endPadding,
          detail::flToAfPadType(type)),
      /* numDims = */ // TODO: check
      std::max(input.ndim(), static_cast<int>(padWidths.size())));
}
} // namespace fl
