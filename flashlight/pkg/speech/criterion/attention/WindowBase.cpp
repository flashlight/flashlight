/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/WindowBase.h"

namespace fl {
namespace app {
namespace asr {

af::array WindowBase::computeInputNotPaddedSize(
    const af::array& inputSizes,
    int inputSteps,
    int batchSize,
    int decoderStepsDim,
    bool doTile) const {
  if (inputSizes.isempty()) {
    if (doTile) {
      return af::constant(
          static_cast<float>(inputSteps),
          af::dim4(decoderStepsDim, inputSteps, batchSize));
    } else {
      return af::constant(
          static_cast<float>(inputSteps), af::dim4(1, 1, batchSize));
    }
  }
  if (inputSizes.elements() != batchSize) {
    throw std::runtime_error(
        "Attention Window: wrong size of the input sizes vector, doesn't match with batchsize");
  }
  af::array inputNotPaddedSize =
      af::ceil(inputSizes / af::max<float>(inputSizes) * inputSteps);
  inputNotPaddedSize =
      af::moddims(inputNotPaddedSize, af::dim4(1, 1, batchSize));
  if (doTile) {
    inputNotPaddedSize =
        af::tile(inputNotPaddedSize, decoderStepsDim, inputSteps, 1);
  }
  return inputNotPaddedSize;
}

af::array WindowBase::computeTargetNotPaddedSize(
    const af::array& targetSizes,
    int inputSteps,
    int targetLen,
    int batchSize,
    int decoderStepsDim) const {
  if (targetSizes.isempty()) {
    return af::constant(
        static_cast<float>(targetLen),
        af::dim4(decoderStepsDim, inputSteps, batchSize));
  }
  if (targetSizes.elements() != batchSize) {
    throw std::runtime_error(
        "Window Attention: wrong size of the target sizes vector, doesn't match with batchsize");
  }
  af::array targetNotPaddedSize = af::moddims(
      af::ceil(targetSizes / af::max<float>(targetSizes) * targetLen),
      1,
      1,
      batchSize);
  targetNotPaddedSize =
      af::tile(targetNotPaddedSize, af::dim4(decoderStepsDim, inputSteps, 1));
  return targetNotPaddedSize;
}

} // namespace asr
} // namespace app
} // namespace fl
