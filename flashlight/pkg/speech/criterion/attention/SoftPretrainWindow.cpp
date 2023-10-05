/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/SoftPretrainWindow.h"

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/pkg/speech/criterion/attention/Defines.h"

namespace fl::pkg::speech {

SoftPretrainWindow::SoftPretrainWindow(double std) : std_(std) {}

Variable SoftPretrainWindow::compute(
    int targetLen,
    int inputSteps,
    int batchSize,
    const Tensor& inputSizes,
    const Tensor& targetSizes,
    Tensor& decoderSteps) const {
  int decoderStepsDim = decoderSteps.dim(0);
  auto ts = fl::arange({decoderStepsDim, inputSteps, batchSize}, 1);
  if (inputSizes.isEmpty() && targetSizes.isEmpty()) {
    return Variable(
        -fl::power(ts - inputSteps / targetLen * decoderSteps, 2) /
            (2 * std_ * std_),
        false);
  }

  Tensor inputNotPaddedSize = computeInputNotPaddedSize(
      inputSizes, inputSteps, batchSize, decoderStepsDim, true);
  Tensor targetNotPaddedSize = computeTargetNotPaddedSize(
      targetSizes, inputSteps, targetLen, batchSize, decoderStepsDim);

  auto maskArray =
      -fl::power(
          ts - inputNotPaddedSize / targetNotPaddedSize * decoderSteps, 2) /
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
    const Tensor& inputSizes,
    const Tensor& targetSizes) const {
  Tensor decoderSteps = fl::full({1, inputSteps, batchSize}, step);
  return compute(
      targetLen, inputSteps, batchSize, inputSizes, targetSizes, decoderSteps);
}

Variable SoftPretrainWindow::computeVectorizedWindow(
    int targetLen,
    int inputSteps,
    int batchSize,
    const Tensor& inputSizes,
    const Tensor& targetSizes) const {
  Tensor decoderSteps = fl::arange({targetLen, inputSteps, batchSize}, 0);
  return compute(
      targetLen, inputSteps, batchSize, inputSizes, targetSizes, decoderSteps);
}
} // namespace fl
