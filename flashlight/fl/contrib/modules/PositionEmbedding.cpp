/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/PositionEmbedding.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

PositionEmbedding::PositionEmbedding(
    int32_t layerDim,
    int32_t maxLen,
    double dropout)
    : dropout_(dropout) {
  auto embeddings = uniform(layerDim, maxLen, -0.1, 0.1, af::dtype::f32, true);
  params_ = {embeddings};
}

std::vector<Variable> PositionEmbedding::forward(
    const std::vector<Variable>& input) {
  int n = input[0].dims(1);
  Variable posEmb =
      tileAs(params_[0].as(input[0].type()).cols(0, n - 1), input[0]);
  if (dropout_ > 0.0 && train_) {
    return {input[0] + dropout(posEmb, dropout_)};
  } else {
    return {input[0] + posEmb};
  }
}

std::vector<Variable> PositionEmbedding::operator()(
    const std::vector<Variable>& input) {
  return forward(input);
}

std::string PositionEmbedding::prettyString() const {
  return "Position Embedding Layer";
}

PositionEmbedding::PositionEmbedding() {}

} // namespace fl
