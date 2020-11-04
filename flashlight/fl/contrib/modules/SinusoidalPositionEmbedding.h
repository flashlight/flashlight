/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/Container.h"
#include "flashlight/fl/nn/modules/LayerNorm.h"
#include "flashlight/fl/nn/modules/Linear.h"
#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/**
 * A module which implements a sinusoidal position embedding layer
 * from "Attention is all you need" https://arxiv.org/pdf/1706.03762.pdf
 *
 * Input dimension at forward is assumed to be CxTxBx1, where
 * C is the number of features (channels),
 * T is the sequence length,
 * B is the batch size.
 *
 * output = input * inputScale + sinPosEmb, where sinPosEmb is a Tensor
 * of dimensions CxTxBx1 computed based on position and C.
 *
 * @param layerDim dimension of the first tensor axis (often features C)
 * @param inputScale scaling parameter of the input before adding
 * sinusoidal embedding to it
 *
 */
class SinusoidalPositionEmbedding : public Container {
 public:
  explicit SinusoidalPositionEmbedding(int32_t layerDim, double inputScale = 1.);
  /**
   * SinusoidalPositionEmbedding::forward(input) expects input[0] to be of
   * dimensions CxTxBx1 with C = layerDim.
   * output[0] = input[0] * inputScale + sinPosEmb, where sinPosEmb is a Tensor
   * of dimensions CxTxBx1 computed based on position and C.
   */
  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::vector<Variable> operator()(const std::vector<Variable>& input);

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Container, layerDim_, inputScale_, scale_)

  int32_t layerDim_;
  double inputScale_;
  af::array scale_;

  SinusoidalPositionEmbedding();
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::SinusoidalPositionEmbedding);
