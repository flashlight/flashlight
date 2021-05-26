/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/nn/Transformer.h"

#include "flashlight/fl/nn/nn.h"

using namespace fl;

namespace {

std::shared_ptr<fl::Linear>
makeTransformerLinear(int inDim, int outDim, float gain = 1.0f) {
  int fanIn = inDim;
  int fanOut = outDim;
  float std = gain * std::sqrt(2.0 / (fanIn + fanOut));
  float bound = std::sqrt(3.0) * std;
  auto w = fl::uniform(outDim, inDim, -bound, bound, f32, true);
  bound = std::sqrt(1.0 / fanIn);
  auto b = fl::uniform(af::dim4(outDim), -bound, bound, af::dtype::f32, true);
  return std::make_shared<fl::Linear>(w, b);
}

std::shared_ptr<fl::Linear>
makeMultiheadedAttentionLinear(int inDim, int outDim, int fanOutMult = 1) {
  int fanIn = inDim;
  int fanOut = outDim * fanOutMult;
  float gain = 1.0;
  float std = gain * std::sqrt(2.0 / (fanIn + fanOut));
  float bound = std::sqrt(3.0) * std;
  auto w = fl::uniform(outDim, inDim, -bound, bound, f32, true);
  auto b = fl::param(af::constant(0, af::dim4(outDim)));
  return std::make_shared<fl::Linear>(w, b);
}

} // namespace

namespace fl {
namespace pkg {
namespace vision {

// query [ E X B X L ]
// values and keys [ E X B X S ]
fl::Variable transformerMultiheadAttention(
    const fl::Variable& query,
    const fl::Variable& key,
    const fl::Variable& value,
    const fl::Variable& keyPaddingMask,
    const int32_t nHead,
    const double pDropout) {
  int32_t bsz = query.dims(1);
  int32_t modelDim = query.dims(0);
  int32_t headDim = modelDim / nHead;
  int32_t tgtLen = query.dims(2);
  int32_t srcLen = key.dims(2);

  auto q = moddims(query, af::dim4(headDim, nHead, bsz, tgtLen));
  auto v = moddims(value, af::dim4(headDim, nHead, bsz, srcLen));
  auto k = moddims(key, af::dim4(headDim, nHead, bsz, srcLen));
  // Reorder so that the "Sequence" is along the first dimension,
  // the embedding is along the zeroth dimension
  q = reorder(q, 0, 3, 1, 2);
  v = reorder(v, 0, 3, 1, 2);
  k = reorder(k, 0, 3, 1, 2);

  auto scores = matmulTN(q, k);

  if (!keyPaddingMask.isempty()) {
    scores = scores +
        tileAs(moddims(log(keyPaddingMask), {1, srcLen, 1, bsz}), scores);
  }

  auto attn = dropout(softmax(scores, 1), pDropout);
  auto result = matmulNT(attn.as(v.type()), v);
  result = moddims(result, af::dim4(tgtLen, modelDim, bsz));
  result = reorder(result, 1, 2, 0);
  return result;
}

MultiheadAttention::MultiheadAttention(
    int32_t modelDim,
    int32_t headDim,
    int32_t numHeads,
    float pDropout)
    : pDropout_(pDropout), numHeads_(numHeads) {
  wq_ = makeMultiheadedAttentionLinear(modelDim, headDim * numHeads, 3);
  wk_ = makeMultiheadedAttentionLinear(modelDim, headDim * numHeads, 3);
  wv_ = makeMultiheadedAttentionLinear(modelDim, headDim * numHeads, 3);
  wf_ = makeMultiheadedAttentionLinear(headDim * numHeads, modelDim);
  add(wq_);
  add(wk_);
  add(wv_);
  add(wf_);
}

std::vector<Variable> MultiheadAttention::forward(
    const Variable& queries,
    const Variable& keys,
    const Variable& values,
    const Variable& keyPaddingMask) {
  assert(queries.dims(0) == keys.dims(0));
  assert(queries.dims(0) == values.dims(0));
  assert(queries.dims(1) == keys.dims(1));
  assert(queries.dims(1) == values.dims(1));
  assert(values.dims(2) == keys.dims(2));

  int32_t modelDim = queries.dims(0);
  int32_t headDim = modelDim / numHeads_;

  if (!keyPaddingMask.isempty()) {
    assert(keyPaddingMask.dims(0) == keys.dims(2));
    assert(keyPaddingMask.dims(1) == keys.dims(1));
  }

  auto q = wq_->forward(queries);
  auto k = wk_->forward(keys);
  auto v = wv_->forward(values);

  q = q / std::sqrt(float(headDim));
  float dropout = train_ ? pDropout_ : 0.0f;
  auto result = transformerMultiheadAttention(
      q, k, v, keyPaddingMask, numHeads_, dropout);
  result = (*wf_)(result);
  assert(result.dims() == queries.dims());
  std::vector<Variable> results = {result};
  return results;
};

std::vector<Variable> MultiheadAttention::forward(
    const std::vector<Variable>& input) {
  assert(input.size() == 4);
  return this->forward(input[0], input[1], input[2], input[3]);
}

std::string MultiheadAttention::prettyString() const {
  std::ostringstream ss;
  ss << "MultiheadAttention";
  ss << Container::prettyString();
  return ss.str();
}

TransformerBaseLayer::TransformerBaseLayer(
    int32_t modelDim,
    int32_t mlpDim,
    int32_t nHeads,
    float pDropout)
    : self_attn_(std::make_shared<MultiheadAttention>(
          modelDim,
          modelDim / nHeads,
          nHeads,
          pDropout)),
      w1_(makeTransformerLinear(modelDim, mlpDim)),
      w2_(makeTransformerLinear(mlpDim, modelDim)),
      norm1_(std::make_shared<LayerNorm>(
          std::vector<int>{0},
          1e-5,
          true,
          modelDim)),
      norm2_(std::make_shared<LayerNorm>(
          std::vector<int>{0},
          1e-5,
          true,
          modelDim)),
      pDropout_(pDropout) {
  add(self_attn_);
  add(w1_);
  add(w2_);
  add(norm1_);
  add(norm2_);
};

Variable TransformerBaseLayer::mlp(const Variable& in) {
  float pDropout = train_ ? pDropout_ : 0.0;
  return (*w2_)(dropout(relu((*w1_)(in)), pDropout));
}

Variable TransformerBaseLayer::withPosEmbed(
    const Variable& input,
    const Variable& pos) {
  if (pos.isempty()) {
    return input;
  }
  return input + pos;
}

Variable TransformerBaseLayer::selfAttention(
    const Variable& input,
    const Variable& pos,
    const Variable& keyPaddingMask) {
  auto k = withPosEmbed(input, pos);
  auto q = withPosEmbed(input, pos);
  return self_attn_->forward(q, k, input, keyPaddingMask)[0];
}

TransformerEncoderLayer::TransformerEncoderLayer(
    int32_t modelDim,
    int32_t mlpDim,
    int32_t nHeads,
    float pDropout)
    : TransformerBaseLayer(modelDim, mlpDim, nHeads, pDropout){};

std::vector<Variable> TransformerEncoderLayer::forward(
    const std::vector<Variable>& input) {
  auto src = input[0];
  auto mask = input[1];
  auto pos = input[2];

  float pDropout = train_ ? pDropout_ : 0.0f;

  auto src2 = this->selfAttention(src, pos, mask);
  src = src + dropout(src2, pDropout);
  src = (*norm1_)(src);
  src2 = mlp(src);
  src = src + dropout(src2, pDropout);
  src = (*norm2_)(src);

  return {src, mask, pos};
}

std::string TransformerEncoderLayer::prettyString() const {
  std::ostringstream ss;
  ss << "TransformerEncoderLayer";
  ss << Container::prettyString();
  return ss.str();
}

TransformerDecoderLayer::TransformerDecoderLayer(
    int32_t modelDim,
    int32_t mlpDim,
    int32_t nHeads,
    float pDropout)
    : self_attn_(std::make_shared<MultiheadAttention>(
          modelDim,
          modelDim / nHeads,
          nHeads,
          pDropout)),
      encoder_attn_(std::make_shared<MultiheadAttention>(
          modelDim,
          modelDim / nHeads,
          nHeads,
          pDropout)),
      w1_(makeTransformerLinear(modelDim, mlpDim)),
      w2_(makeTransformerLinear(mlpDim, modelDim)),
      norm1_(std::make_shared<LayerNorm>(
          std::vector<int>{0},
          1e-5,
          true,
          modelDim)),
      norm2_(std::make_shared<LayerNorm>(
          std::vector<int>{0},
          1e-5,
          true,
          modelDim)),
      norm3_(std::make_shared<LayerNorm>(
          std::vector<int>{0},
          1e-5,
          true,
          modelDim)),
      pDropout_(pDropout) {
  add(self_attn_);
  add(encoder_attn_);
  add(w1_);
  add(w2_);
  add(norm1_);
  add(norm2_);
  add(norm3_);
}

Variable TransformerDecoderLayer::mlp(const Variable& in) {
  float pDropout = train_ ? pDropout_ : 0.0;
  return (*w2_)(dropout(relu((*w1_)(in)), pDropout));
}

Variable TransformerDecoderLayer::withPosEmbed(
    const Variable& input,
    const Variable& pos) {
  if (pos.isempty()) {
    return input;
  }
  return input + pos;
}

Variable TransformerDecoderLayer::selfAttention(
    const Variable& input,
    const Variable& pos,
    const Variable& keyPaddingMask /* = Variable() */) {
  auto k = withPosEmbed(input, pos);
  auto q = withPosEmbed(input, pos);
  return self_attn_->forward(q, k, input, keyPaddingMask)[0];
}

std::vector<Variable> TransformerDecoderLayer::forward(
    const std::vector<Variable>& input) {
  auto tgt = input[0];
  auto memory = input[1];
  auto pos = (input.size() > 2) ? input[2] : Variable();
  auto queryPos = (input.size() > 3) ? input[3] : Variable();
  auto memoryKeyPaddingMask = (input.size() > 4) ? input[4] : Variable();

  float pDropout = train_ ? pDropout_ : 0.0f;

  auto tgt2 = this->selfAttention(tgt, queryPos);
  tgt = tgt + dropout(tgt2, pDropout);
  tgt = (*norm1_)(tgt);
  tgt2 = encoder_attn_->forward({
      this->withPosEmbed(tgt, queryPos), // queries
      this->withPosEmbed(memory, pos), // keys
      memory, // values
      memoryKeyPaddingMask // mask
  })[0];
  tgt = tgt + dropout(tgt2, pDropout);
  tgt = (*norm2_)(tgt);
  tgt2 = mlp(tgt);
  tgt = tgt + dropout(tgt2, pDropout);
  tgt = (*norm3_)(tgt);
  return {tgt};
}

std::string TransformerDecoderLayer::prettyString() const {
  std::ostringstream ss;
  ss << "TransformerDecoderLayer";
  ss << Container::prettyString();
  return ss.str();
}

TransformerDecoder::TransformerDecoder(
    int32_t modelDim,
    int32_t mlpDim,
    int32_t nHeads,
    int32_t layers,
    float pDropout) {
  // TODO add norm
  for (int i = 0; i < layers; i++) {
    add(TransformerDecoderLayer(modelDim, mlpDim, nHeads, pDropout));
  }
  add(LayerNorm(std::vector<int>{0}, 1e-5, true, modelDim));
}

std::vector<Variable> TransformerDecoder::forward(
    const std::vector<Variable>& input) {
  auto tgt = input[0];
  auto memory = input[1];
  auto pos = (input.size() > 2) ? input[2] : Variable();
  auto query_pos = (input.size() > 3) ? input[3] : Variable();
  auto mask = (input.size() > 4) ? input[4] : Variable();

  fl::Variable output = tgt;
  auto mods = modules();

  std::vector<Variable> intermediate;
  for (int i = 0; i < mods.size() - 1; i++) {
    output = mods[i]->forward({output, memory, pos, query_pos, mask})[0];
    intermediate.push_back(mods.back()->forward({output})[0]);
  }
  return {concatenate(intermediate, 3)};
}
std::string TransformerDecoder::prettyString() const {
  std::ostringstream ss;
  ss << "TransformerDecoder";
  ss << Container::prettyString();
  return ss.str();
}

TransformerEncoder::TransformerEncoder(
    int32_t modelDim,
    int32_t mlpDim,
    int32_t nHeads,
    int32_t layers,
    float pDropout) {
  for (int i = 0; i < layers; i++) {
    add(TransformerEncoderLayer(modelDim, mlpDim, nHeads, pDropout));
  }
}

std::vector<Variable> TransformerEncoder::forward(
    const std::vector<Variable>& input) {
  std::vector<Variable> output = input;
  auto mods = modules();
  for (int i = 0; i < mods.size(); i++) {
    output = mods[i]->forward(output);
  }
  return output;
}

std::string TransformerEncoder::prettyString() const {
  std::ostringstream ss;
  ss << "TransformerDecoder";
  ss << Container::prettyString();
  return ss.str();
}

Transformer::Transformer(
    int32_t modelDim,
    int32_t numHeads,
    int32_t numEncoderLayers,
    int32_t numDecoderLayers,
    int32_t mlpDim,
    float pDropout)
    : encoder_(std::make_shared<TransformerEncoder>(
          modelDim,
          mlpDim,
          numHeads,
          numEncoderLayers,
          pDropout)),
      decoder_(std::make_shared<TransformerDecoder>(
          modelDim,
          mlpDim,
          numHeads,
          numDecoderLayers,
          pDropout)) {
  add(encoder_);
  add(decoder_);
};

std::vector<Variable> Transformer::forward(
    Variable src,
    Variable mask,
    Variable queryEmbed,
    Variable posEmbed) {
  assert(src.dims(2) == queryEmbed.dims(0));

  int B = src.dims(3);
  // Reshape from [ W X H X C X B ] to [ WH X C X B ]
  src = flatten(src, 0, 1);
  // Flatten to C x B x WH
  src = reorder(src, 1, 2, 0);

  posEmbed = flatten(posEmbed, 0, 1);
  posEmbed = reorder(posEmbed, 1, 2, 0);

  mask = flatten(mask, 0, 2);

  // Tile object queries for each batch
  af::dim4 unsqueeze = {queryEmbed.dims(0), 1, queryEmbed.dims(1)};
  queryEmbed = moddims(queryEmbed, unsqueeze);
  queryEmbed = tile(queryEmbed, {1, B, 1});
  assert(queryEmbed.dims(1) == src.dims(1));
  assert(queryEmbed.dims(0) == src.dims(0));

  auto tgt =
      fl::Variable(af::constant(0, queryEmbed.dims(), src.type()), false);

  auto memory = encoder_->forward({src, mask, posEmbed});
  auto hs = decoder_->forward({tgt, memory[0], posEmbed, queryEmbed, mask})[0];

  auto reordered = reorder(hs, 0, 2, 1);
  return {reordered};
}

std::vector<Variable> Transformer::forward(const std::vector<Variable>& input) {
  assert(input.size() > 3);
  auto src = input[0];
  auto mask = (input.size() > 1) ? input[1] : Variable();
  auto query_embed = (input.size() > 2) ? input[2] : Variable();
  auto pos_embed = (input.size() > 3) ? input[3] : Variable();
  return forward(src, mask, query_embed, pos_embed);
}

std::string Transformer::prettyString() const {
  std::ostringstream ss;
  ss << "Transformer";
  ss << Container::prettyString();
  return ss.str();
}

} // namespace vision
} // namespace pkg
} // namespace fl
