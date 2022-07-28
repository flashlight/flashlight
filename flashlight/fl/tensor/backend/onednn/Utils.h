/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"

#include <dnnl.hpp>

namespace fl {
namespace detail {

/**
 * Convert an fl::dtype into a OneDNN dnnl::memory::data_type
 *
 * @param[in] type a Flashlight data type.
 * @return the corresponding OneDNN memory data type.
 * @throw invalid_argument if corresponding type doesn't exist.
 */
dnnl::memory::data_type flToOneDnnType(const fl::dtype type);

/**
 * Convert a OneDNN dnnl::memory::data_type into an fl::dtype
 *
 * @param[in] type a OneDNN memory data type.
 * @return the corresponding Flashlight data type.
 * @throw invalid_argument if corresponding type doesn't exist.
 */
fl::dtype oneDnnToFlType(const dnnl::memory::data_type type);

/**
 * Return whether the given Flashlight type has a corresponding OneDNN memory
 * type.
 *
 * @param[in] shape a Flashlight shape.
 * @return true if given type has a corresponding OneDNN memory type.
 */
bool isTypeSupportedByOneDnn(const fl::dtype type);

/**
 * Get a string representation of given OneDNN memory data type.
 *
 * @return a string reprenseting given OneDNN memory data type.
 */
std::string oneDnnDataTypeToStr(const dnnl::memory::data_type type);

/**
 * Convert a Flashlight Shape to OneDNN dimensions.
 *
 * @return the corresponding OneDNN dims for given shape.
 */
dnnl::memory::dims shapeToOneDnnDims(const Shape& shape);

/**
 * Convert a Flashlight Shape to OneDNN strides, assuming row-major order.
 *
 * @return the corresponding OneDNN strides for given shape.
 */
dnnl::memory::dims shapeToOneDnnStrides(const Shape& shape);

/**
 * Return the input type that can represent a larger range of data.
 *
 * @param[in] t1 the first input type.
 * @param[in] t2 the second input type.
 * @return the input type that can represent a larger range of data.
 */
dnnl::memory::data_type getTypeWithLargerRange(
    dnnl::memory::data_type t1,
    dnnl::memory::data_type t2);

} // namespace detail
} // namespace fl
