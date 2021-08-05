/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/SinusoidalPositionEmbedding.h"

#include <math.h>
#include <stdexcept>
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

SinusoidalPositionEmbedding::SinusoidalPositionEmbedding(
    int32_t layerDim,
    double inputScale /* = 1. */)
    : layerDim_(layerDim), inputScale_(inputScale) {

  // 'scale_' chosen based on positional embedding from:
  //      Attention is All You Need
  //      Ashish Vaswani, et al. (2017)
  //      https://arxiv.org/pdf/1706.03762.pdf
  // Create an `iota` that looks like `[0,0,1,1,2,2,...]`, use it as the scale.
  scale_ = af::exp(-2 * af::floor((af::iota(layerDim_) / 2)) * std::log(10000) / layerDim_);
  // Create a Cosine phase shift that acts on indices like `[0,1,0,1,...]`
  const double sinToCosPhaseShift = af::Pi / 2.0;
  cosShifts_ = sinToCosPhaseShift * af::iota(layerDim_) % 2;
  // In the forward pass, the even indices of embedding vectors will have the Sine
  // function applied and the odd indices will have the Cosine function applied.
}

std::vector<Variable> SinusoidalPositionEmbedding::forward(
  const std::vector<Variable>& input) {
  if (input[0].dims(0) != layerDim_) {
    throw std::invalid_argument(
        "Input dimenstion " + std::to_string(input[0].dims(0)) +
        " and Embedding dimension " + std::to_string(layerDim_) +
        " are different.");
  }
  // Retrieve the number of tokens (positions) and the numeric type (floating point precision).
  const int nPositions = input[0].dims(1);
  const auto numType = input[0].type();
  // Generate the tensor of positions for each token vector [embedding size, num positions].
  //  positions = [[ 0,  0, ..],
  //               [ 1,  1, ..],
  //               [.., .., ..]]
  af::array positions = af::iota(af::dim4(1, nPositions), af::dim4(layerDim_), numType);
  // Generate the embedding transformation with the precomputed scale and shift factors.
  positions = af::sin(positions * af::tile(scale_.as(numType), 1, nPositions)
                                + af::tile(cosShifts_.as(numType), 1, nPositions));
  // Convert the positional embedding into a variable (for gradient tracking).
  Variable embeddingsPos = Variable(positions, false);
  // Return the inputs with the positional embeddings tiled over the batch dimension.
  return {input[0] * inputScale_ + tileAs(embeddingsPos, input[0])};
}

std::vector<Variable> SinusoidalPositionEmbedding::operator()(
    const std::vector<Variable>& input) {
  return forward(input);
}

std::string SinusoidalPositionEmbedding::prettyString() const {
  std::ostringstream ss;
  ss << "Sinusoidal Position Embedding Layer (embDim: " << layerDim_
     << "), (input scale " << inputScale_ << ")";
  return ss.str();
}

SinusoidalPositionEmbedding::SinusoidalPositionEmbedding() {}

} // namespace fl
