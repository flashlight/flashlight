/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/tensor/VisionOps.h"

#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/pkg/vision/tensor/VisionExtension.h"
#include "flashlight/pkg/vision/tensor/VisionExtensionBackends.h"

namespace fl {

Tensor histogram(
    const Tensor& tensor,
    const unsigned numBins,
    const double minVal,
    const double maxVal) {
  return tensor.backend().getExtension<VisionExtension>().histogram(
      tensor, numBins, minVal, maxVal);
}

Tensor histogram(const Tensor& tensor, const unsigned numBins) {
  return tensor.backend().getExtension<VisionExtension>().histogram(
      tensor, numBins);
}

Tensor equalize(const Tensor& input, const Tensor& histogram) {
  return input.backend().getExtension<VisionExtension>().equalize(
      input, histogram);
}

Tensor resize(
    const Tensor& tensor,
    const Shape& shape,
    const InterpolationMode mode /* = InterpolationMode::Nearest */) {
  return tensor.backend().getExtension<VisionExtension>().resize(
      tensor, shape, mode);
}

Tensor rotate(
    const Tensor& input,
    const float theta,
    const Tensor& fill /* = Tensor() */) {
  return input.backend().getExtension<VisionExtension>().rotate(
      input, theta, fill);
}

Tensor rotate(
    const Tensor& input,
    const float theta,
    const InterpolationMode mode /* = InterpolationMode::Nearest */) {
  return input.backend().getExtension<VisionExtension>().rotate(
      input, theta, mode);
}

Tensor translate(
    const Tensor& input,
    const Shape& translation,
    const Shape& outputDims /* = {} */,
    const Tensor& fill /* = Tensor() */) {
  return input.backend().getExtension<VisionExtension>().translate(
      input, translation, outputDims, fill);
}

Tensor translate(
    const Tensor& input,
    const Shape& translation,
    const Shape& outputDims /* = {} */,
    const InterpolationMode mode /* = InterpolationMode::Nearest */) {
  return input.backend().getExtension<VisionExtension>().translate(
      input, translation, outputDims, mode);
}

Tensor shear(
    const Tensor& input,
    const std::vector<float>& skews,
    const Shape& outputDims /* = {} */,
    const Tensor& fill /* = Tensor() */) {
  return input.backend().getExtension<VisionExtension>().shear(
      input, skews, outputDims, fill);
}

Tensor shear(
    const Tensor& input,
    const std::vector<float>& skews,
    const Shape& outputDims /* = {} */,
    const InterpolationMode mode /* = InterpolationMode::Nearest */) {
  return input.backend().getExtension<VisionExtension>().shear(
      input, skews, outputDims, mode);
}

Tensor gaussianFilter(const Shape& shape) {
  // TODO{fl::Tensor} - empty tensor instantiation for default backend
  return defaultTensorBackend().getExtension<VisionExtension>().gaussianFilter(
      shape);
}

} // namespace fl
