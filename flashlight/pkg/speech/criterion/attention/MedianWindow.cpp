/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/MedianWindow.h"

#include <stdexcept>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/pkg/speech/criterion/attention/Defines.h"

namespace fl::pkg::speech {

MedianWindow::MedianWindow() = default;
MedianWindow::MedianWindow(int wL, int wR) : wL_(wL), wR_(wR) {}

Variable MedianWindow::computeWindow(
    const Variable& prevAttn, // [1, windowsize, batchSize]
    int step,
    int targetLen,
    int inputSteps,
    int batchSize,
    const Tensor& inputSizes,
    const Tensor& targetSizes) const {
  // Each row of prevAttn is the attention for an input utterance.
  // The attention vector is output from a softmax.
  // The definition of "median" is the point where cdf passes 0.5.

  int width = std::min(wL_ + wR_, inputSteps);
  Tensor inputNotPaddedSize =
      computeInputNotPaddedSize(inputSizes, inputSteps, batchSize, 0, false);

  if (step == 0 || width == inputSteps) {
    // [1, inputSteps]
    auto maskArray = fl::full({1, inputSteps, batchSize}, 0.0);
    maskArray(fl::span, fl::range(0, width), fl::span) = 1.0;
    auto indicesAdd = fl::arange({1, inputSteps, batchSize}, 1);
    maskArray(indicesAdd >= fl::tile(inputNotPaddedSize, {1, inputSteps})) =
        0.0;
    // [1, inputSteps, batchSize]
    return Variable(fl::log(maskArray), false);
  }

  auto mIdx =
      fl::sum(
          fl::cumsum(prevAttn.tensor(), 1) < 0.5, {1}, /* keepDims = */ true)
          .astype(fl::dtype::s32);
  auto startIdx = mIdx - wL_;

  // check boundary conditions and adjust the window
  auto startDiff = fl::abs(fl::clip(startIdx, -wL_, 0));
  startIdx = startIdx + startDiff;

  auto endDiff = fl::abs(
      fl::clip(startIdx + wL_ + wR_ - inputNotPaddedSize, 0, wL_ + wR_));
  startIdx = startIdx - endDiff;

  auto maskArray = fl::full({1, inputSteps, batchSize}, 0.0, fl::dtype::f32);
  auto indices = fl::arange({width, batchSize}, 0) +
      fl::tile(fl::reshape(startIdx, {1, batchSize}), {width, 1}) +
      fl::tile(fl::reshape(
                   fl::arange(0, batchSize * inputSteps, inputSteps),
                   {1, batchSize}),
               {width, 1});
  maskArray(indices.flatten()) = 1.0;
  auto indicesAdd = fl::arange({1, inputSteps, batchSize}, 1);
  maskArray(indicesAdd >= fl::tile(inputNotPaddedSize, {1, inputSteps})) = 0.0;

  if (!targetSizes.isEmpty()) {
    Tensor targetNotPaddedSize = computeTargetNotPaddedSize(
        targetSizes, inputSteps, targetLen, batchSize, 1);
    maskArray(step >= targetNotPaddedSize) = 0.0;
  }
  maskArray = fl::log(maskArray);
  // force all -inf values to be kAttentionMaskValue to avoid nan in softmax
  maskArray(maskArray < kAttentionMaskValue) = kAttentionMaskValue;
  // [1, inputSteps, batchSize]
  return Variable(maskArray, false);
}

Variable MedianWindow::computeVectorizedWindow(
    int /* unused */,
    int /* unused */,
    int /* unused */,
    const Tensor& /* unused */,
    const Tensor& /* unused */) const {
  throw std::invalid_argument(
      "MedianWindow does not support vectorized window mask");
}
} // namespace fl
