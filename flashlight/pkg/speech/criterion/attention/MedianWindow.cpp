/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/MedianWindow.h"
#include "flashlight/pkg/speech/criterion/attention/Defines.h"

namespace fl {
namespace pkg {
namespace speech {

MedianWindow::MedianWindow() {}
MedianWindow::MedianWindow(int wL, int wR) : wL_(wL), wR_(wR) {}

Variable MedianWindow::computeWindow(
    const Variable& prevAttn, // [1, windowsize, batchSize]
    int step,
    int targetLen,
    int inputSteps,
    int batchSize,
    const af::array& inputSizes,
    const af::array& targetSizes) const {
  // Each row of prevAttn is the attention for an input utterance.
  // The attention vector is output from a softmax.
  // The definition of "median" is the point where cdf passes 0.5.

  int width = std::min(wL_ + wR_, inputSteps);
  af::array inputNotPaddedSize =
      computeInputNotPaddedSize(inputSizes, inputSteps, batchSize, 0, false);

  if (step == 0 || width == inputSteps) {
    // [1, inputSteps]
    auto maskArray = af::constant(0.0, af::dim4(1, inputSteps, batchSize));
    maskArray(af::span, af::seq(0, width - 1), af::span) = 1.0;
    auto indicesAdd = af::range(af::dim4(1, inputSteps, batchSize), 1);
    maskArray(
        indicesAdd >= af::tile(inputNotPaddedSize, af::dim4(1, inputSteps))) =
        0.0;
    // [1, inputSteps, batchSize]
    return Variable(af::log(maskArray), false);
  }

  auto mIdx =
      af::sum(af::accum(prevAttn.array(), 1) < 0.5, 1).as(af::dtype::s32);
  auto startIdx = mIdx - wL_;

  // check boundary conditions and adjust the window
  auto startDiff = af::abs(af::clamp(startIdx, -wL_, 0));
  startIdx = startIdx + startDiff;
  auto endDiff = af::abs(
      af::clamp(startIdx + wL_ + wR_ - inputNotPaddedSize, 0, wL_ + wR_));
  startIdx = startIdx - endDiff;

  auto maskArray = af::constant(0.0, 1, inputSteps, batchSize, f32);
  auto indices = af::range(af::dim4(width, batchSize), 0) +
      af::tile(af::moddims(startIdx, {1, batchSize}), {width, 1}) +
      af::tile(af::moddims(
                   af::seq(0, batchSize * inputSteps - 1, inputSteps),
                   {1, batchSize}),
               {width, 1});
  maskArray(af::flat(indices)) = 1.0;
  auto indicesAdd = af::range(af::dim4(1, inputSteps, batchSize), 1);
  maskArray(
      indicesAdd >= af::tile(inputNotPaddedSize, af::dim4(1, inputSteps))) =
      0.0;

  if (!targetSizes.isempty()) {
    af::array targetNotPaddedSize = computeTargetNotPaddedSize(
        targetSizes, inputSteps, targetLen, batchSize, 1);
    maskArray(step >= targetNotPaddedSize) = 0.0;
  }
  maskArray = af::log(maskArray);
  // force all -inf values to be kAttentionMaskValue to avoid nan in softmax
  maskArray(maskArray < kAttentionMaskValue) = kAttentionMaskValue;
  // [1, inputSteps, batchSize]
  return Variable(maskArray, false);
}

Variable MedianWindow::computeVectorizedWindow(
    int /* unused */,
    int /* unused */,
    int /* unused */,
    const af::array& /* unused */,
    const af::array& /* unused */) const {
  throw af::exception("MedianWindow does not support vectorized window mask");
}
} // namespace speech
} // namespace pkg
} // namespace fl
