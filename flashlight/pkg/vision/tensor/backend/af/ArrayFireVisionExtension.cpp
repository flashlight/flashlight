/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/tensor/backend/af/ArrayFireVisionExtension.h"

#include <stdexcept>

#include <af/data.h>
#include <af/image.h>

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"
#include "flashlight/pkg/vision/tensor/VisionOps.h"

namespace fl {
namespace detail {

constexpr af_interp_type flToAfInterpType(InterpolationMode mode) {
  switch (mode) {
    case InterpolationMode::Nearest:
      return AF_INTERP_NEAREST;
    case InterpolationMode::Linear:
      return AF_INTERP_LINEAR;
    case InterpolationMode::Bilinear:
      return AF_INTERP_BILINEAR;
    case InterpolationMode::Cubic:
      return AF_INTERP_CUBIC;
    case InterpolationMode::Bicubic:
      return AF_INTERP_BICUBIC;
    default:
      throw std::invalid_argument(
          "flToAfInterpType - no corresponding ArrayFire "
          "interpolation mode for given interpolation mode.");
  }
}

namespace {

/*
 * Performs a fill image operation using a fill Tensor on an input Tensor in
 * conjunction with some image transform.
 *
 * This is needed because ArrayFire only supports zero-filling on empty spots.
 * Once AF supports filling directly, this can be removed.
 */
template <typename af_image_transform_func_t, typename... Args>
af::array addFillTensor(
    const af::array& input,
    const af::array& fillImg,
    af_image_transform_func_t transformFunc,
    Args&&... args) {
  af::array res = input;

  const double delta = 1e-2;
  if (!fillImg.isempty()) {
    res = res + delta;
  }

  // Call the transform
  res = transformFunc(res, std::forward<Args>(args)...);

  if (!fillImg.isempty()) {
    auto mask = af::sum(res, 2) == 0;
    mask = af::tile(mask, {1, 1, 3});
    res = mask * fillImg + (1 - mask) * (res - delta);
  }
  return res;
}

} // namespace
} // namespace detail

bool ArrayFireVisionExtension::isDataTypeSupported(
    const fl::dtype& dtype) const {
  return ArrayFireBackend::getInstance().isDataTypeSupported(dtype);
}

Tensor ArrayFireVisionExtension::histogram(
    const Tensor& tensor,
    const unsigned numBins,
    const double minVal,
    const double maxVal) {
  // TODO: add ndim to this
  return toTensor<ArrayFireTensor>(
      af::histogram(toArray(tensor), numBins, minVal, maxVal),
      /* numDims = */ 1);
}

Tensor ArrayFireVisionExtension::histogram(
    const Tensor& tensor,
    const unsigned numBins) {
  // TODO: add ndim to this
  return toTensor<ArrayFireTensor>(
      af::histogram(toArray(tensor), numBins), /* numDims = */ 1);
}

Tensor ArrayFireVisionExtension::equalize(
    const Tensor& input,
    const Tensor& histogram) {
  return toTensor<ArrayFireTensor>(
      af::histEqual(toArray(input), toArray(histogram)), input.ndim());
}

Tensor ArrayFireVisionExtension::resize(
    const Tensor& tensor,
    const Shape& shape,
    const InterpolationMode mode) {
  af::dim4 _shape = detail::flToAfDims(shape);
  return toTensor<ArrayFireTensor>(
      af::resize(
          toArray(tensor),
          _shape[0],
          _shape[1],
          detail::flToAfInterpType(mode)),
      tensor.ndim());
}

Tensor ArrayFireVisionExtension::rotate(
    const Tensor& input,
    const float theta,
    const Tensor& fill /* = Tensor() */) {
  return toTensor<ArrayFireTensor>(
      detail::addFillTensor(
          toArray(input),
          toArray(fill),
          af::rotate,
          theta,
          /* crop = */ true,
          AF_INTERP_NEAREST),
      input.ndim());
}

Tensor ArrayFireVisionExtension::rotate(
    const Tensor& input,
    const float theta,
    const InterpolationMode mode) {
  return toTensor<ArrayFireTensor>(
      af::rotate(toArray(input), theta, detail::flToAfInterpType(mode)),
      input.ndim());
}

Tensor ArrayFireVisionExtension::translate(
    const Tensor& input,
    const Shape& translation,
    const Shape& outputDimsIn /* = {} */,
    const Tensor& fill /* = Tensor() */) {
  // If no output dims specified, AF expects 2D 0's which to discard OOB data
  Shape outputDims = outputDimsIn;
  if (outputDimsIn.ndim() == 0) {
    outputDims = Shape({0, 0});
  }

  if (translation.ndim() != 2 || outputDims.ndim() != 2) {
    throw std::invalid_argument(
        "ArrayFireVisionExtension::shear - "
        "only 2D skews shapes and empty or 2D output shapes are supported");
  }

  return toTensor<ArrayFireTensor>(
      detail::addFillTensor(
          toArray(input),
          toArray(fill),
          af::translate,
          translation[0],
          translation[1],
          outputDims[0],
          outputDims[1],
          AF_INTERP_NEAREST),
      input.ndim());
}

Tensor ArrayFireVisionExtension::translate(
    const Tensor& input,
    const Shape& translation,
    const Shape& outputDimsIn /* = {} */,
    const InterpolationMode mode) {
  // If no output dims specified, AF expects 2D 0's which to discard OOB data
  Shape outputDims = outputDimsIn;
  if (outputDimsIn.ndim() == 0) {
    outputDims = Shape({0, 0});
  }

  if (translation.ndim() != 2 || outputDims.ndim() != 2) {
    throw std::invalid_argument(
        "ArrayFireVisionExtension::shear - "
        "only 2D skews shapes and empty or 2D output shapes are supported");
  }

  af::dim4 _translations = detail::flToAfDims(translation);
  af::dim4 _outputDims = detail::flToAfInterpType(mode);

  return toTensor<ArrayFireTensor>(
      af::translate(
          toArray(input),
          _translations[0],
          _translations[1],
          _outputDims[0],
          _outputDims[1]),
      input.ndim());
}

Tensor ArrayFireVisionExtension::shear(
    const Tensor& input,
    const std::vector<float>& skews,
    const Shape& outputDimsIn /* = {} */,
    const Tensor& fill /* = Tensor() */) {
  // If no output dims specified, AF expects 2D 0's which to discard OOB data
  Shape outputDims = outputDimsIn;
  if (outputDimsIn.ndim() == 0) {
    outputDims = Shape({0, 0});
  }

  if (skews.size() != 2 || outputDims.ndim() != 2) {
    throw std::invalid_argument(
        "ArrayFireVisionExtension::shear - "
        "only 2D skews shapes and empty or 2D output shapes are supported");
  }

  af::dim4 _outputDims = detail::flToAfDims(outputDims);

  return toTensor<ArrayFireTensor>(
      detail::addFillTensor(
          toArray(input),
          toArray(fill),
          af::skew,
          skews[0],
          skews[1],
          _outputDims[0],
          _outputDims[1],
          /* inverse = */ true,
          AF_INTERP_NEAREST),
      input.ndim());
}

Tensor ArrayFireVisionExtension::shear(
    const Tensor& input,
    const std::vector<float>& skews,
    const Shape& outputDimsIn /* = {} */,
    const InterpolationMode mode) {
  // If no output dims specified, AF expects 2D 0's which to discard OOB data
  Shape outputDims = outputDimsIn;
  if (outputDimsIn.ndim() == 0) {
    outputDims = Shape({0, 0});
  }

  if (skews.size() != 2 || outputDims.ndim() != 2) {
    throw std::invalid_argument(
        "ArrayFireVisionExtension::shear - "
        "only 2D skews shapes and empty or 2D output shapes are supported");
  }

  return toTensor<ArrayFireTensor>(
      af::skew(
          toArray(input),
          skews[0],
          skews[1],
          outputDims[0],
          outputDims[1],
          /* inverse = */ true,
          detail::flToAfInterpType(mode)),
      input.ndim());
}

Tensor ArrayFireVisionExtension::gaussianFilter(const Shape& shape) {
  af::dim4 _shape = detail::flToAfDims(shape);
  return toTensor<ArrayFireTensor>(
      af::gaussianKernel(_shape[0], _shape[1]), shape.ndim());
}

} // namespace fl
