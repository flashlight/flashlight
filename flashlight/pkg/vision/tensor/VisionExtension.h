/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/TensorExtension.h"
#include "flashlight/pkg/vision/tensor/VisionOps.h"

namespace fl {

// TODO: rename this file to VisionExtension
class VisionExtension : public TensorExtension<VisionExtension> {
 public:
  static constexpr TensorExtensionType extensionType =
      TensorExtensionType::Vision;

  VisionExtension() = default;
  virtual ~VisionExtension() = default;

  virtual Tensor histogram(
      const Tensor& tensor,
      const unsigned numBins,
      const double minVal,
      const double maxVal) = 0;
  virtual Tensor histogram(const Tensor& tensor, const unsigned numBins) = 0;

  virtual Tensor equalize(const Tensor& input, const Tensor& histogram) = 0;

  virtual Tensor resize(
      const Tensor& tensor,
      const Shape& shape,
      const InterpolationMode mode) = 0;

  virtual Tensor
  rotate(const Tensor& input, const float theta, const Tensor& fill) = 0;
  virtual Tensor rotate(
      const Tensor& input,
      const float theta,
      const InterpolationMode mode) = 0;

  virtual Tensor translate(
      const Tensor& input,
      const Shape& translation,
      const Shape& outputDims,
      const Tensor& fill) = 0;
  virtual Tensor translate(
      const Tensor& input,
      const Shape& translation,
      const Shape& outputDims,
      const InterpolationMode mode) = 0;

  virtual Tensor shear(
      const Tensor& input,
      const std::vector<float>& skews,
      const Shape& outputDims,
      const Tensor& fill) = 0;
  virtual Tensor shear(
      const Tensor& input,
      const std::vector<float>& skews,
      const Shape& outputDims,
      const InterpolationMode mode) = 0;

  virtual Tensor gaussianFilter(const Shape& shape) = 0;
};

} // namespace fl
