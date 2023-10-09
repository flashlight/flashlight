/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/SinusoidalPositionEmbedding.h"

#include <algorithm>
#include <cmath>
#include <numeric>
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
  scale_ = fl::exp(
      -2 * fl::floor((fl::iota({layerDim_}) / 2)) * std::log(10000) /
      layerDim_);
  // Create a Cosine phase shift that acts on indices like `[0,1,0,1,...]`
  const double sinToCosPhaseShift = M_PI / 2.0;
  cosShifts_ = sinToCosPhaseShift * fl::iota({layerDim_}) % 2;
  // In the forward pass, the even indices of embedding vectors will have the
  // Sine function applied and the odd indices will have the Cosine function
  // applied.
}

SinusoidalPositionEmbedding::SinusoidalPositionEmbedding(
    const SinusoidalPositionEmbedding& other)
    : layerDim_(other.layerDim_),
      inputScale_(other.inputScale_),
      scale_(other.scale_.copy()),
      cosShifts_(other.cosShifts_.copy()) {}

SinusoidalPositionEmbedding& SinusoidalPositionEmbedding::operator=(
    const SinusoidalPositionEmbedding& other) {
  layerDim_ = other.layerDim_;
  inputScale_ = other.inputScale_;
  scale_ = other.scale_.copy();
  cosShifts_ = other.cosShifts_.copy();
  return *this;
}

std::vector<Variable> SinusoidalPositionEmbedding::forward(
    const std::vector<Variable>& input) {
  if (input[0].dim(0) != layerDim_) {
    throw std::invalid_argument(
        "Input dimenstion " + std::to_string(input[0].dim(0)) +
        " and Embedding dimension " + std::to_string(layerDim_) +
        " are different.");
  }
  // Retrieve the number of tokens (positions) and the numeric type (floating
  // point precision).
  const int nPositions = input[0].dim(1);
  const auto numType = input[0].type();
  // Generate the tensor of positions for each token vector [embedding size, num
  // positions].
  //  positions = [[ 0,  0, ..],
  //               [ 1,  1, ..],
  //               [.., .., ..]]
  Tensor positions = fl::iota({1, nPositions}, {layerDim_}, numType);
  // Generate the embedding transformation with the precomputed scale and shift
  // factors.
  positions = fl::sin(
      positions * fl::tile(scale_.astype(numType), {1, nPositions}) +
      fl::tile(cosShifts_.astype(numType), {1, nPositions}));
  // Convert the positional embedding into a variable (for gradient tracking).
  Variable embeddingsPos = Variable(positions, false);
  // Return the inputs with the positional embeddings tiled over the batch
  // dimension.
  return {input[0] * inputScale_ + tileAs(embeddingsPos, input[0])};
}

std::vector<Variable> SinusoidalPositionEmbedding::operator()(
    const std::vector<Variable>& input) {
  return forward(input);
}

std::unique_ptr<Module> SinusoidalPositionEmbedding::clone() const {
  return std::make_unique<SinusoidalPositionEmbedding>(*this);
}

std::string SinusoidalPositionEmbedding::prettyString() const {
  std::ostringstream ss;
  ss << "Sinusoidal Position Embedding Layer (embDim: " << layerDim_
     << "), (input scale " << inputScale_ << ")";
  return ss.str();
}

SinusoidalPositionEmbedding::SinusoidalPositionEmbedding() = default;

} // namespace fl
