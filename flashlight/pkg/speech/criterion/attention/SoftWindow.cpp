/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/SoftWindow.h"
#include "flashlight/pkg/speech/criterion/attention/Defines.h"

namespace fl {
namespace app {
namespace asr {

SoftWindow::SoftWindow() {}
SoftWindow::SoftWindow(double std, double avgRate, int offset)
    : std_(std), avgRate_(avgRate), offset_(offset) {}

Variable SoftWindow::compute(
    int targetLen,
    int inputSteps,
    int batchSize,
    const af::array& inputSizes,
    const af::array& targetSizes,
    af::array& decoderSteps) const {
  int decoderStepsDim = decoderSteps.dims(0);
  auto ts = af::range(af::dim4(decoderStepsDim, inputSteps, batchSize), 1);
  af::array inputNotPaddedSize = computeInputNotPaddedSize(
      inputSizes, inputSteps, batchSize, decoderStepsDim, true);

  af::array centers = af::round(af::min(
      offset_ + decoderSteps * avgRate_, inputNotPaddedSize - avgRate_));
  auto maskArray = -af::pow(ts - centers, 2) / (2 * std_ * std_);
  maskArray(ts >= inputNotPaddedSize) = -std::numeric_limits<float>::infinity();

  if (!targetSizes.isempty()) {
    af::array targetNotPaddedSize = computeTargetNotPaddedSize(
        targetSizes, inputSteps, targetLen, batchSize, decoderStepsDim);
    maskArray(decoderSteps >= targetNotPaddedSize) = kAttentionMaskValue;
  }
  // [decoderStepsDim, inputSteps, batchSize]
  return Variable(maskArray, false);
}

Variable SoftWindow::computeWindow(
    const Variable& /* unused */,
    int step,
    int targetLen,
    int inputSteps,
    int batchSize,
    const af::array& inputSizes,
    const af::array& targetSizes) const {
  af::array decoderSteps =
      af::constant(step, af::dim4(1, inputSteps, batchSize));
  return compute(
      targetLen, inputSteps, batchSize, inputSizes, targetSizes, decoderSteps);
}

Variable SoftWindow::computeVectorizedWindow(
    int targetLen,
    int inputSteps,
    int batchSize,
    const af::array& inputSizes,
    const af::array& targetSizes) const {
  af::array decoderSteps =
      af::range(af::dim4(targetLen, inputSteps, batchSize), 0);
  return compute(
      targetLen, inputSteps, batchSize, inputSizes, targetSizes, decoderSteps);
}
} // namespace asr
} // namespace app
} // namespace fl
