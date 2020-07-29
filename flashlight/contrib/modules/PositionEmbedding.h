/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/nn/modules/Container.h"
#include "flashlight/nn/modules/LayerNorm.h"
#include "flashlight/nn/modules/Linear.h"
#include "flashlight/nn/modules/Module.h"

namespace fl {

/**
 * A module which implements a position embedding layer
 *
 * Input dimension at forward is assumed to be CxTxBx1, where
 * C is the number of features (channels),
 * T is the sequence length (smaller than maxLen),
 * B is the batch size.
 *
 */
class PositionEmbedding : public Container {
 public:
  PositionEmbedding(int32_t layerDim, int32_t maxLen, double dropout = 0);

  /**
   * PositionEmbedding::forward(input) expects input[0] to be of
   * dimensions CxTxBx1 with C = layerDim and T <= maxLen.
   * output[0] = input[0] + pos_emb, where pos_emb is a Tensor of dimensions
   * CxTxBx1, and pos_emb = this.param_[0][:T], so pos_emb will be randomly
   * initialized absolute position embeddings, that can be learned end-to-end.
   *
   */
  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::vector<Variable> operator()(const std::vector<Variable>& input);

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Container, dropout_)

  double dropout_;

  PositionEmbedding();
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::PositionEmbedding);
