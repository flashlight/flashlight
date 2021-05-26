/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/StepWindow.h"
#include "flashlight/pkg/speech/criterion/attention/Defines.h"

namespace fl {
namespace pkg {
namespace speech {

StepWindow::StepWindow() {}
StepWindow::StepWindow(int sMin, int sMax, double vMin, double vMax)
    : sMin_(sMin), sMax_(sMax), vMin_(vMin), vMax_(vMax) {}

Variable StepWindow::compute(
    int targetLen,
    int inputSteps,
    int batchSize,
    const af::array& inputSizes,
    const af::array& targetSizes,
    af::array& decoderSteps) const {
  int decoderStepsDim = decoderSteps.dims(0);
  af::array inputNotPaddedSize = computeInputNotPaddedSize(
      inputSizes, inputSteps, batchSize, decoderStepsDim, true);
  auto startIdx = af::max(
      0,
      af::round(
          af::min(inputNotPaddedSize - vMax_, sMin_ + decoderSteps * vMin_))
          .as(s32));
  auto endIdx = af::min(
      inputNotPaddedSize, af::round(sMax_ + decoderSteps * vMax_).as(s32));
  af::array indices = af::iota(
      af::dim4(1, inputSteps, 1), af::dim4(decoderStepsDim, 1, batchSize));

  // [decoderStepsDim, inputSteps, batchSize]
  auto maskArray =
      af::constant(1.0, af::dim4(decoderStepsDim, inputSteps, batchSize));
  maskArray(indices < startIdx) = 0.0;
  maskArray(indices >= endIdx) = 0.0;
  if (!targetSizes.isempty()) {
    af::array targetNotPaddedSize = computeTargetNotPaddedSize(
        targetSizes, inputSteps, targetLen, batchSize, decoderStepsDim);
    maskArray(decoderSteps >= targetNotPaddedSize) = 0.0;
  }
  // force all -inf values to be kAttentionMaskValue to avoid nan in softmax
  maskArray = af::log(maskArray);
  maskArray(maskArray < kAttentionMaskValue) = kAttentionMaskValue;
  return Variable(maskArray, false);
}

Variable StepWindow::computeWindow(
    const Variable& /* unused */,
    int step,
    int targetLen,
    int inputSteps,
    int batchSize,
    const af::array& inputSizes,
    const af::array& targetSizes) const {
  auto decoderSteps = af::constant(step, af::dim4(1, inputSteps, batchSize));
  return compute(
      targetLen, inputSteps, batchSize, inputSizes, targetSizes, decoderSteps);
}

Variable StepWindow::computeVectorizedWindow(
    int targetLen,
    int inputSteps,
    int batchSize,
    const af::array& inputSizes,
    const af::array& targetSizes) const {
  auto decoderSteps =
      iota(af::dim4(targetLen), af::dim4(1, inputSteps, batchSize));
  return compute(
      targetLen, inputSteps, batchSize, inputSizes, targetSizes, decoderSteps);
}
} // namespace speech
} // namespace pkg
} // namespace fl
