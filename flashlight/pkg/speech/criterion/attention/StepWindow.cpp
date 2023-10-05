/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/StepWindow.h"

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/pkg/speech/criterion/attention/Defines.h"

namespace fl::pkg::speech {

StepWindow::StepWindow() = default;
StepWindow::StepWindow(int sMin, int sMax, double vMin, double vMax)
    : sMin_(sMin), sMax_(sMax), vMin_(vMin), vMax_(vMax) {}

Variable StepWindow::compute(
    int targetLen,
    int inputSteps,
    int batchSize,
    const Tensor& inputSizes,
    const Tensor& targetSizes,
    Tensor& decoderSteps) const {
  int decoderStepsDim = decoderSteps.dim(0);
  Tensor inputNotPaddedSize = computeInputNotPaddedSize(
      inputSizes, inputSteps, batchSize, decoderStepsDim, true);
  Tensor startIdx = fl::maximum(
      0,
      fl::rint(
          fl::minimum(inputNotPaddedSize - vMax_, sMin_ + decoderSteps * vMin_))
          .astype(fl::dtype::s32));
  auto endIdx = fl::minimum(
      inputNotPaddedSize,
      fl::rint(sMax_ + decoderSteps * vMax_).astype(fl::dtype::s32));
  Tensor indices =
      fl::iota({1, inputSteps, 1}, {decoderStepsDim, 1, batchSize});

  // [decoderStepsDim, inputSteps, batchSize]
  Tensor maskTensor = fl::full({decoderStepsDim, inputSteps, batchSize}, 1.0);
  maskTensor(indices < startIdx) = 0.0;
  maskTensor(indices >= endIdx) = 0.0;
  if (!targetSizes.isEmpty()) {
    Tensor targetNotPaddedSize = computeTargetNotPaddedSize(
        targetSizes, inputSteps, targetLen, batchSize, decoderStepsDim);
    maskTensor(decoderSteps >= targetNotPaddedSize) = 0.0;
  }
  // force all -inf values to be kAttentionMaskValue to avoid nan in softmax
  maskTensor = fl::log(maskTensor);
  maskTensor(maskTensor < kAttentionMaskValue) = kAttentionMaskValue;
  return Variable(maskTensor, false);
}

Variable StepWindow::computeWindow(
    const Variable& /* unused */,
    int step,
    int targetLen,
    int inputSteps,
    int batchSize,
    const Tensor& inputSizes,
    const Tensor& targetSizes) const {
  auto decoderSteps = fl::full({1, inputSteps, batchSize}, step);
  return compute(
      targetLen, inputSteps, batchSize, inputSizes, targetSizes, decoderSteps);
}

Variable StepWindow::computeVectorizedWindow(
    int targetLen,
    int inputSteps,
    int batchSize,
    const Tensor& inputSizes,
    const Tensor& targetSizes) const {
  auto decoderSteps = fl::iota({targetLen}, {1, inputSteps, batchSize});
  return compute(
      targetLen, inputSteps, batchSize, inputSizes, targetSizes, decoderSteps);
}
} // namespace fl
