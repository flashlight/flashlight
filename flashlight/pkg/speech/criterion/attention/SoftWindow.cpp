/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/SoftWindow.h"

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/pkg/speech/criterion/attention/Defines.h"

namespace fl::pkg::speech {

SoftWindow::SoftWindow() = default;
SoftWindow::SoftWindow(double std, double avgRate, int offset)
    : std_(std), avgRate_(avgRate), offset_(offset) {}

Variable SoftWindow::compute(
    int targetLen,
    int inputSteps,
    int batchSize,
    const Tensor& inputSizes,
    const Tensor& targetSizes,
    Tensor& decoderSteps) const {
  int decoderStepsDim = decoderSteps.dim(0);
  auto ts = fl::arange({decoderStepsDim, inputSteps, batchSize}, 1);
  Tensor inputNotPaddedSize = computeInputNotPaddedSize(
      inputSizes, inputSteps, batchSize, decoderStepsDim, true);

  Tensor centers = fl::rint(fl::minimum(
      offset_ + decoderSteps * avgRate_, inputNotPaddedSize - avgRate_));
  auto maskArray = -fl::power(ts - centers, 2) / (2 * std_ * std_);
  maskArray(ts >= inputNotPaddedSize) = -std::numeric_limits<float>::infinity();

  if (!targetSizes.isEmpty()) {
    Tensor targetNotPaddedSize = computeTargetNotPaddedSize(
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
    const Tensor& inputSizes,
    const Tensor& targetSizes) const {
  Tensor decoderSteps = fl::full({1, inputSteps, batchSize}, step);
  return compute(
      targetLen, inputSteps, batchSize, inputSizes, targetSizes, decoderSteps);
}

Variable SoftWindow::computeVectorizedWindow(
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
