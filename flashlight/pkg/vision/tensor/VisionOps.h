/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/**
 * Compute a histogram based on some input tensor values.
 *
 * @param[in] tensor the input tensor
 * @param[in] numBins the number of bins between min and max to populate
 * @param[in] minVal (optional) the minimum bin value. If not specified, this is
 * set to the minimum value present in the input.
 * @param[in] maxVal (optional) the maximum bin value. If not specified, this is
 * set to the maximum value present in the input.
 * @return a 1D Tensor of dimensions {num bins} containing histogram counts.
 */
Tensor histogram(
    const Tensor& tensor,
    const unsigned numBins,
    const double minVal,
    const double maxVal);
Tensor histogram(const Tensor& tensor, const unsigned numBins);

/**
 * Perform histogram equalization on an input tensor.
 *
 * @param[in] input the input tensor
 * @param[in] histogram the histogram tensor: expects a 1D tensor across bin
 * side with each bin
 * @return a Tensor with the histogram equalization operation applied.
 */
Tensor equalize(const Tensor& input, const Tensor& histogram);

/**
 * Modes for performing interpolation operations on image transformations.
 *
 * TODO{fl::Tensor} -- consider moving this to a more general place - other
 * things will need to support interpolation
 */
enum class InterpolationMode { Nearest, Linear, Bilinear, Cubic, Bicubic };

/**
 * Resize a tensor, performing interpolation as needed.
 *
 * @param[in] input the input tensor
 * @param[in] shape the shape to which to resize the tensor
 * @param[in] mode the interpolation mode to use to fill in unfilled or new
 * image regions after the operation is applied
 * @return a Tensor with the resize operation applied.
 */
Tensor resize(
    const Tensor& tensor,
    const Shape& shape,
    const InterpolationMode mode = InterpolationMode::Nearest);

/**
 * Rotate a tensor by a given angle, filling in unfilled locations as needed.
 *
 * @param[in] input the input tensor
 * @param[in] theta the angle by which to rotate the input image
 * @param[in] fill a tensor with shape {3} (across channels) used to fill in
 * empty image regions post-transformation
 * @return a Tensor with the rotation operation applied.
 */
Tensor
rotate(const Tensor& input, const float theta, const Tensor& fill = Tensor());

/**
 * Rotate a tensor by a given angle, filling in unfilled locations as needed.
 *
 * @param[in] input the input tensor
 * @param[in] theta the angle by which to rotate the input image
 * @param[in] mode the interpolation mode to use to fill in unfilled image
 * regions after the operation is applied
 * @return a Tensor with the rotation operation applied.
 */
Tensor rotate(
    const Tensor& input,
    const float theta,
    const InterpolationMode mode = InterpolationMode::Nearest);

/**
 * Translate a tensor by given amounts along some axes, filling in unfilled
 * locations as specified.
 *
 * @param[in] input the input tensor
 * @param[in] translation the translation to make across each dimension
 * @param[in] outputDims the shape of the output tensor. If empty, the output
 * dimensions are the same as the input dims, and out-of-bounds data is
 * discarded.
 * @param[in] fill a tensor with shape {3} (across channels) used to fill in
 * empty image regions post-transformation
 * @return a Tensor with the translation operation applied.
 */
Tensor translate(
    const Tensor& input,
    const Shape& translation,
    const Shape& outputDims = {},
    const Tensor& fill = Tensor());

/**
 * Translate a tensor by given amounts along some axes, filling in unfilled
 * locations via interpolation.
 *
 * @param[in] input the input tensor
 * @param[in] translation the translation to make across each dimension
 * @param[in] outputDims the shape of the output tensor. If empty, the output
 * dimensions are the same as the input dims, and out-of-bounds data is
 * discarded.
 * @param[in] mode the interpolation mode to use to fill in unfilled image
 * regions after the operation is applied
 * @return a Tensor with the translation operation applied.
 */
Tensor translate(
    const Tensor& input,
    const Shape& translation,
    const Shape& outputDims = {},
    const InterpolationMode mode = InterpolationMode::Nearest);

/**
 * Apply a shear transformation (also called a skew transformation) to an input
 * image, filling unfilled/new image elements with elements from a fill tensor.
 *
 * @param[in] input the input tensor
 * @param[in] skews a shape containing the skew along each axis
 * @param[in] outputDims the shape of the output tensor. If empty, the output
 * dimensions are the same as the input dims, and out-of-bounds data is
 * discarded.
 * @param[in] fill a tensor with shape {3} (across channels) used to fill in
 * empty image regions post-transformation
 * @return a Tensor with the shear operation applied.
 */
Tensor shear(
    const Tensor& input,
    const std::vector<float>& skews,
    const Shape& outputDims = {},
    const Tensor& fill = Tensor());

/**
 * Apply a shear transformation (also called a skew transformation) to an input
 * image, interpolating unfilled elements.
 *
 * @param[in] input the input tensor
 * @param[in] skews a shape containing the skew along each axis
 * @param[in] outputDims the shape of the output tensor. If empty, the output
 * dimensions are the same as the input dims, and out-of-bounds data is
 * discarded.
 * @param[in] mode the interpolation mode to use to fill in unfilled image
 * regions after the operation is applied
 * @return a Tensor with the shear operation applied.
 */
Tensor shear(
    const Tensor& input,
    const std::vector<float>& skews,
    const Shape& outputDims = {},
    const InterpolationMode mode = InterpolationMode::Nearest);

/**
 * Create a Tensor with the given shape that is Gaussian distributed across the
 * tensor dimensions.
 *
 * @param[in] shape the shape of the Gaussian filter
 * @return a Tensor of the specified shape with the given Gaussian filter
 */
Tensor gaussianFilter(const Shape& shape);

} // namespace fl
