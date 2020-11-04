/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
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
  int32_t halfDim = layerDim_ / 2;
  if (halfDim % 2 == 1) {
    halfDim++;
  }
  // size is layerDim_ / 2 + 1 x 1 x 1 x 1
  scale_ =
      af::exp(-2 * af::iota(af::dim4(halfDim)) * std::log(10000) / layerDim_);
}

std::vector<Variable> SinusoidalPositionEmbedding::forward(
    const std::vector<Variable>& input) {
  if (input[0].dims(0) != layerDim_) {
    throw std::invalid_argument(
        "Input dimenstion " + std::to_string(input[0].dims(0)) +
        " and Embedding dimension " + std::to_string(layerDim_) +
        " are different");
  }
  int nPositions = input[0].dims(1);
  af::array positions =
      af::iota(af::dim4(1, nPositions), af::dim4(scale_.dims(0), 1));
  int32_t halfDim = layerDim_ / 2;
  af::array positionsScaledSin = positions * af::tile(scale_, 1, nPositions);
  af::array positionsScaledCos =
      positions * af::tile(scale_(af::seq(halfDim)), 1, nPositions);
  Variable sinPos = Variable(af::sin(positionsScaledSin), false);
  Variable cosPos = Variable(af::cos(positionsScaledCos), false);

  Variable embeddingsPos = fl::concatenate({sinPos, cosPos}, 0).as(input[0].type());
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
