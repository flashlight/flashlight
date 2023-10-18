/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/decoder/ConvLmModule.h"

#include <stdexcept>
#include <string>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl::pkg::speech {

GetConvLmScoreFunc buildGetConvLmScoreFunction(
    std::shared_ptr<Module> network) {
  auto getConvLmScoreFunc = [network](
                                const std::vector<int>& inputs,
                                const std::vector<int>& lastTokenPositions,
                                int sampleSize = -1,
                                int batchSize = 1) {
    sampleSize = sampleSize > 0 ? sampleSize : inputs.size();
    if (sampleSize * batchSize > inputs.size()) {
      throw std::invalid_argument(
          "[ConvLM] Incorrect sample size (" + std::to_string(sampleSize) +
          ") or batch size (" + std::to_string(batchSize) + ").");
    }
    Tensor inputData = Tensor::fromVector({sampleSize, batchSize}, inputs);
    fl::Variable output = network->forward({fl::input(inputData)})[0];

    if (fl::countNonzero(fl::isnan(output.tensor())).asScalar<int>() != 0) {
      throw std::runtime_error("[ConvLM] Encountered NaNs in propagation");
    }
    int32_t C = output.dim(0), T = output.dim(1), B = output.dim(2);
    if (B != batchSize) {
      throw std::logic_error(
          "[ConvLM]: incorrect predictions: batch should be " +
          std::to_string(batchSize) + " but it is " + std::to_string(B));
    }
    if (batchSize != (int)lastTokenPositions.size()) {
      throw std::logic_error(
          "[ConvLM]: incorrect postions for accessing: size should be " +
          std::to_string(batchSize) + " but it is " +
          std::to_string(lastTokenPositions.size()));
    }
    // output (c, t, b)
    // set global indices: offset by channel
    Tensor globalIndices = fl::iota({C, 1}, {1, B}, fl::dtype::s32);
    // set global indices: offset by batch
    globalIndices =
        globalIndices + fl::iota({1, B}, {C, 1}, fl::dtype::s32) * T * C;
    // set global indices: offset by time which we need to take
    globalIndices = globalIndices +
        fl::tile(Tensor::fromVector({1, B}, lastTokenPositions), {C, 1}) * C;
    Tensor preds =
        fl::reshape(output.tensor().flatten()(globalIndices.flatten()), {C, B});
    // vector of B X C predictions
    return preds.toHostVector<float>();
  };

  return getConvLmScoreFunc;
}
} // namespace fl
