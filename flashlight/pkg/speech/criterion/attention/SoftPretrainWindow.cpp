/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/SoftPretrainWindow.h"
#include "flashlight/pkg/speech/criterion/attention/Defines.h"

namespace fl {
namespace pkg {
namespace speech {

SoftPretrainWindow::SoftPretrainWindow(double std) : std_(std) {}

Variable SoftPretrainWindow::compute(
    int targetLen,
    int inputSteps,
    int batchSize,
    const af::array& inputSizes,
    const af::array& targetSizes,
    af::array& decoderSteps) const {
  int decoderStepsDim = decoderSteps.dims(0);
  auto ts = af::range(af::dim4(decoderStepsDim, inputSteps, batchSize), 1);
  if (inputSizes.isempty() && targetSizes.isempty()) {
    return Variable(
        -af::pow(ts - inputSteps / targetLen * decoderSteps, 2) /
            (2 * std_ * std_),
        false);
  }

  af::array inputNotPaddedSize = computeInputNotPaddedSize(
      inputSizes, inputSteps, batchSize, decoderStepsDim, true);
  af::array targetNotPaddedSize = computeTargetNotPaddedSize(
      targetSizes, inputSteps, targetLen, batchSize, decoderStepsDim);

  auto maskArray =
      -af::pow(ts - inputNotPaddedSize / targetNotPaddedSize * decoderSteps, 2) /
      (2 * std_ * std_);
  maskArray(ts >= inputNotPaddedSize) = -std::numeric_limits<float>::infinity();
  maskArray(decoderSteps >= targetNotPaddedSize) =
      -std::numeric_limits<float>::infinity();
  // force all -inf values to be kAttentionMaskValue to avoid nan in softmax
  maskArray(maskArray < kAttentionMaskValue) = kAttentionMaskValue;
  // [decoderStepsDim, inputSteps, batchSize]
  return Variable(maskArray, false);
}

Variable SoftPretrainWindow::computeWindow(
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

Variable SoftPretrainWindow::computeVectorizedWindow(
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
} // namespace speech
} // namespace pkg
} // namespace fl
