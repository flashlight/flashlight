/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/Container.h"
#include "flashlight/fl/nn/modules/LayerNorm.h"
#include "flashlight/fl/nn/modules/Linear.h"
#include "flashlight/fl/nn/modules/Module.h"

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
class FL_API PositionEmbedding : public Module {
 public:
  PositionEmbedding(int32_t layerDim, int32_t maxLen, double dropout = 0);

  PositionEmbedding(const PositionEmbedding& other);

  PositionEmbedding& operator=(const PositionEmbedding& other);

  PositionEmbedding(PositionEmbedding&& other) = default;

  PositionEmbedding& operator=(PositionEmbedding&& other) = default;

  /**
   * PositionEmbedding::forward(input) expects input[0] to be of
   * dimensions C x T x B with C = layerDim and T <= maxLen.
   *
   * output[0] = input[0] + pos_emb, where pos_emb is a Tensor of dimensions
   * C x T x B, and pos_emb = this.param_[0][:T], so pos_emb will be randomly
   * initialized absolute position embeddings, that can be learned end-to-end.
   *
   */
  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::vector<Variable> operator()(const std::vector<Variable>& input);

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Module, dropout_)

  double dropout_;

  PositionEmbedding();
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::PositionEmbedding);
