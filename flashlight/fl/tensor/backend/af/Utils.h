/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/data.h>

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"

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

} // namespace detail
} // namespace fl
