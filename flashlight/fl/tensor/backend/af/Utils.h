/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/data.h>
#include <af/index.h>
#include <af/seq.h>

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"

namespace fl {

class Index; // see Index.h
class range; // see Index.h

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
 * Convert an fl::Shape into an ArrayFire af::dim4
 */
af::dim4 flToAfDims(const Shape& shape);

/**
 * Convert an ArrayFire af::dim4 into an fl::Shape
 */
Shape afToFlDims(const af::dim4& d);

/**
 * Convert an ArrayFire af::dim4 into an fl::Shape, in-place
 */
void afToFlDims(const af::dim4& d, Shape& s);

/**
 * Convert an fl::range into an af::seq.
 */
af::seq flRangeToAfSeq(const fl::range& range);

/**
 * Convert an fl::Index into an af::index.
 */
af::index flToAfIndex(const fl::Index& idx);

/**
 * Modify the dimensions (in place via af::moddims) or an Array to have no 1
 * indices. For example, an Array of shape (1, 2, 1, 6) becomes (2, 6).
 *
 * This operation is performed after indexing, etc where the resulting ArrayFire
 * shape would have 1's in it.
 */
af::array condenseIndices(const af::array& arr);

} // namespace detail
} // namespace fl
