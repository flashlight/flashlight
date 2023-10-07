/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/vision/tensor/VisionExtension.h"

#include <af/image.h>

namespace fl {
namespace detail {

/*
 * Convert a Flashlight interpolation mode into an ArrayFire interpolation mode
 */
constexpr af_interp_type flToAfInterpType(InterpolationMode mode);

} // namespace detail

class ArrayFireVisionExtension : public VisionExtension {
 public:
  bool isDataTypeSupported(const fl::dtype& dtype) const override;

  Tensor histogram(
      const Tensor& tensor,
      const unsigned numBins,
      const double minVal,
      const double maxVal) override;
  Tensor histogram(const Tensor& tensor, const unsigned numBins) override;

  Tensor equalize(const Tensor& input, const Tensor& histogram) override;

  Tensor resize(
      const Tensor& tensor,
      const Shape& shape,
      const InterpolationMode mode) override;

  Tensor rotate(const Tensor& input, const float theta, const Tensor& fill)
      override;
  Tensor rotate(
      const Tensor& input,
      const float theta,
      const InterpolationMode mode) override;

  Tensor translate(
      const Tensor& input,
      const Shape& translation,
      const Shape& outputDims,
      const Tensor& fill) override;
  Tensor translate(
      const Tensor& input,
      const Shape& translation,
      const Shape& outputDims,
      const InterpolationMode mode) override;

  Tensor shear(
      const Tensor& input,
      const std::vector<float>& skews,
      const Shape& outputDims,
      const Tensor& fill) override;
  Tensor shear(
      const Tensor& input,
      const std::vector<float>& skews,
      const Shape& outputDims,
      const InterpolationMode mode) override;

  Tensor gaussianFilter(const Shape& shape) override;
};

} // namespace fl
