/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/data.h>
#include <af/exception.h>
#include <af/index.h>
#include <af/seq.h>

#include <vector>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/Types.h"

#define AF_CHECK(fn)                                                          \
  do {                                                                        \
    af_err __err = fn;                                                        \
    if (__err == AF_SUCCESS) {                                                \
      break;                                                                  \
    }                                                                         \
    throw af::exception(                                                      \
        "ArrayFire error: ", __PRETTY_FUNCTION__, __FILE__, __LINE__, __err); \
  } while (0)

namespace fl {
namespace detail {

/**
 * Convert an fl::dtype into an ArrayFire af::dtype
 */
af::dtype flToAfType(fl::dtype type);

/**
 * Convert an ArrayFire af::dtype into an fl::dtype
 */
fl::dtype afToFlType(af::dtype type);

/**
 * Convert a Flashlight matrix property into an ArrayFire matrix property.
 */
af_mat_prop flToAfMatrixProperty(MatrixProperty property);

/**
 * Convert a Flashlight tensor storage type into an ArrayFire storage type.
 */
af_storage flToAfStorageType(StorageType storageType);

/**
 * Convert a Flashlight tensor sort mode into an ArrayFire topk sort mode.
 */
af_topk_function flToAfTopKSortMode(SortMode sortMode);

/**
 * Convert an fl::Shape into an ArrayFire af::dim4
 */
af::dim4 flToAfDims(const Shape& shape);

/**
 * Convert an ArrayFire af::dim4 into an fl::Shape
 */
Shape afToFlDims(const af::dim4& d, const unsigned numDims);

/**
 * Convert an ArrayFire af::dim4 into an fl::Shape, in-place
 */
void afToFlDims(const af::dim4& d, const unsigned numDims, Shape& s);

/**
 * Convert an fl::range into an af::seq.
 */
af::seq flRangeToAfSeq(const fl::range& range);

/**
 * Convert an fl::Index into an af::index.
 */
af::index flToAfIndex(const fl::Index& idx);

std::vector<af::index> flToAfIndices(const std::vector<fl::Index>& flIndices);

/**
 * Strip leading 1 indices from an ArrayFire dim4.
 */
af::dim4 condenseDims(const af::dim4& dims);

/**
 * Modify the dimensions (in place via af::moddims) or an Array to have no 1
 * indices. For example, an Array of shape (1, 2, 1, 6) becomes (2, 6).
 *
 * This operation is performed before returning Array shape, etc where the
 * resulting ArrayFire shape would have 1's in it.
 *
 * If keepDims is true, this is a noop, and the array is returned as is.
 */
af::array condenseIndices(
    const af::array& arr,
    const bool keepDims = false,
    const std::optional<std::vector<detail::IndexType>>& indexTypes = {},
    const bool isFlat = false);

/**
 * Convert a Flashlight Location into an ArrayFire location (host or device).
 */
af_source flToAfLocation(Location location);

/**
 * Construct an ArrayFire array from a buffer and Flashlight details.
 */
af::array fromFlData(
    const Shape& shape,
    const void* ptr,
    fl::dtype type,
    fl::Location memoryLocation);

/**
 * Convert a Flashlight PadType to an ArrayFire af_border_type for describing
 * padding.
 */
af_border_type flToAfPadType(PadType type);

} // namespace detail
} // namespace fl
