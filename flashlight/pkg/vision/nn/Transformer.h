/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>

#include "flashlight/pkg/vision/dataset/BoxUtils.h"
#include "flashlight/fl/nn/nn.h"

namespace fl {
namespace app {
namespace objdet {

fl::Variable transformerMultiheadAttention(
    const fl::Variable& query,
    const fl::Variable& key,
    const fl::Variable& value,
    const fl::Variable& keyPaddingMask,
    const int32_t nHead,
    const double pDropout);

class MultiheadAttention : public Container {
 public:
  MultiheadAttention(
      int32_t modelDim,
      int32_t headDim,
      int32_t numHeads,
      float pDropout = 0.f);

  // queries [ E, N, L ], where L is target length, N is batch size.
  // keys / values  [ E, N, S ], where S is src length, N is batch size.
  // keyPaddingMask [ S, N ]
  std::vector<Variable> forward(
      const Variable& queries,
      const Variable& keys,
      const Variable& values,
      const Variable& keyPaddingMask);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::string prettyString() const override;

 protected:
  std::shared_ptr<Linear> wq_;
  std::shared_ptr<Linear> wk_;
  std::shared_ptr<Linear> wv_;
  std::shared_ptr<Linear> wf_;
  float pDropout_;
  int32_t numHeads_;

 private:
  MultiheadAttention() = default;
  FL_SAVE_LOAD_WITH_BASE(
      fl::Container,
      pDropout_,
      numHeads_,
      wq_,
      wk_,
      wv_,
      wf_)
};

class TransformerBaseLayer : public Container {
 public:
  TransformerBaseLayer(
      int32_t modelDim,
      int32_t mlpDim,
      int32_t nHeads,
      float pDropout);

 protected:
  TransformerBaseLayer() = default;
  std::shared_ptr<MultiheadAttention> self_attn_;
  std::shared_ptr<Linear> w1_, w2_;
  std::shared_ptr<LayerNorm> norm1_, norm2_;
  float pDropout_;

  Variable mlp(const Variable& in);

  Variable withPosEmbed(const Variable& input, const Variable& pos);

  Variable selfAttention(
      const Variable& input,
      const Variable& pos,
      const Variable& keyPaddingMask = Variable());

 private:
  FL_SAVE_LOAD_WITH_BASE(
      fl::Container,
      pDropout_,
      self_attn_,
      w1_,
      w2_,
      norm1_,
      norm2_)
};

class TransformerEncoderLayer : public TransformerBaseLayer {
 public:
  TransformerEncoderLayer(
      int32_t modelDim,
      int32_t mlpDim,
      int32_t nHeads,
      float pDropout);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::string prettyString() const override;

 private:
  TransformerEncoderLayer() = default;
  FL_SAVE_LOAD_WITH_BASE(TransformerBaseLayer)
};

class TransformerDecoderLayer : public Container {
 public:
  TransformerDecoderLayer(
      int32_t modelDim,
      int32_t mlpDim,
      int32_t nHeads,
      float pDropout);

 protected:
  Variable mlp(const Variable& in);

  Variable withPosEmbed(const Variable& input, const Variable& pos);
  Variable selfAttention(
      const Variable& input,
      const Variable& pos,
      const Variable& keyPaddingMask = Variable());

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::string prettyString() const override;

 private:
  TransformerDecoderLayer() = default;
  std::shared_ptr<MultiheadAttention> self_attn_, encoder_attn_;
  std::shared_ptr<Linear> w1_, w2_;
  std::shared_ptr<LayerNorm> norm1_, norm2_, norm3_;
  float pDropout_;
  FL_SAVE_LOAD_WITH_BASE(
      fl::Container,
      pDropout_,
      self_attn_,
      encoder_attn_,
      w1_,
      w2_,
      norm1_,
      norm2_,
      norm3_)
};

class TransformerDecoder : public Container {
 public:
  TransformerDecoder(
      int32_t modelDim,
      int32_t mlpDim,
      int32_t nHeads,
      int32_t layers,
      float pDropout);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::string prettyString() const override;

 private:
  TransformerDecoder() = default;
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};

class TransformerEncoder : public Container {
 public:
  TransformerEncoder(
      int32_t modelDim,
      int32_t mlpDim,
      int32_t nHeads,
      int32_t layers,
      float pDropout);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::string prettyString() const override;

 private:
  TransformerEncoder() = default;
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};

class Transformer : public Container {
 public:
  Transformer(
      int32_t modelDim,
      int32_t numHeads,
      int32_t numEncoderLayers,
      int32_t numDecoderLayers,
      int32_t mlpDim,
      float pDropout);

  /*
   * We expect src to be [ W X H X C X B ]
   * mask to be [ W X H X 1 X B ]
   * query embed [ C X N ] (where N is number of query vectors)
   * and posEmbed to be [ W X H X C X B ]
   * where C is modelDim, B is Batch size, and W and H are width and height of
   * image
   */
  std::vector<Variable>
  forward(Variable src, Variable mask, Variable queryEmbed, Variable posEmbed);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::string prettyString() const override;

 private:
  Transformer() = default;
  std::shared_ptr<TransformerEncoder> encoder_;
  std::shared_ptr<TransformerDecoder> decoder_;
  FL_SAVE_LOAD_WITH_BASE(fl::Container, encoder_, decoder_)
};

} // namespace objdet
} // namespace app
} // namespace fl
CEREAL_REGISTER_TYPE(fl::app::objdet::Transformer)
CEREAL_REGISTER_TYPE(fl::app::objdet::MultiheadAttention)
CEREAL_REGISTER_TYPE(fl::app::objdet::TransformerEncoder)
CEREAL_REGISTER_TYPE(fl::app::objdet::TransformerEncoderLayer)
CEREAL_REGISTER_TYPE(fl::app::objdet::TransformerDecoder)
CEREAL_REGISTER_TYPE(fl::app::objdet::TransformerDecoderLayer)
