/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/WindowBase.h"

namespace fl::pkg::speech {

Tensor WindowBase::computeInputNotPaddedSize(
    const Tensor& inputSizes,
    int inputSteps,
    int batchSize,
    int decoderStepsDim,
    bool doTile) const {
  if (inputSizes.isEmpty()) {
    if (doTile) {
      return fl::full(
          {decoderStepsDim, inputSteps, batchSize}, inputSteps, fl::dtype::f32);
    } else {
      return fl::full({1, 1, batchSize}, inputSteps, fl::dtype::f32);
    }
  }
  if (inputSizes.elements() != batchSize) {
    throw std::runtime_error(
        "Attention Window: wrong size of the input sizes vector, doesn't match with batchsize");
  }
  Tensor inputNotPaddedSize = fl::ceil(
      inputSizes / fl::amax(inputSizes).asScalar<float>() * inputSteps);
  inputNotPaddedSize = fl::reshape(inputNotPaddedSize, {1, 1, batchSize});
  if (doTile) {
    inputNotPaddedSize =
        fl::tile(inputNotPaddedSize, {decoderStepsDim, inputSteps, 1});
  }
  return inputNotPaddedSize;
}

Tensor WindowBase::computeTargetNotPaddedSize(
    const Tensor& targetSizes,
    int inputSteps,
    int targetLen,
    int batchSize,
    int decoderStepsDim) const {
  if (targetSizes.isEmpty()) {
    return fl::full(
        {decoderStepsDim, inputSteps, batchSize}, targetLen, fl::dtype::f32);
  }
  if (targetSizes.elements() != batchSize) {
    throw std::runtime_error(
        "Window Attention: wrong size of the target sizes vector, doesn't match with batchsize");
  }
  Tensor targetNotPaddedSize = fl::reshape(
      fl::ceil(
          targetSizes / fl::amax(targetSizes).asScalar<float>() * targetLen),
      {1, 1, batchSize});
  targetNotPaddedSize =
      fl::tile(targetNotPaddedSize, {decoderStepsDim, inputSteps, 1});
  return targetNotPaddedSize;
}

} // namespace fl
