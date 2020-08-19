/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/contrib/modules/Transformer.h"
#include "flashlight/autograd/Functions.h"
#include "flashlight/nn/Init.h"
#include "flashlight/nn/Utils.h"

namespace {
fl::Variable transformerInitLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std, af::dtype::f32, true);
}

fl::Variable transformerRotate(const fl::Variable& input) {
  auto data = input.array();
  int d0 = data.dims(0);
  int d1 = data.dims(1);
  int d2 = data.dims(2);
  int d3 = data.dims(3);
  data = af::join(0, data, af::constant(0.0, d1, d1, d2, d3));
  data = af::moddims(data, af::dim4((d0 + d1) * d1, 1, d2, d3));
  data = data.rows(0, (d1 + d0 - 1) * d1 - 1);
  data = af::moddims(data, af::dim4(d0 + d1 - 1, d1, d2, d3));
  auto gradFunc = [d0, d1, d2, d3](
                      std::vector<fl::Variable>& inputs,
                      const fl::Variable& gradOutput) {
    auto gradData = gradOutput.array();
    gradData = af::moddims(gradData, af::dim4((d0 + d1 - 1) * d1, 1, d2, d3));
    gradData = af::join(0, gradData, af::constant(0.0, d1, 1, d2, d3));
    gradData = af::moddims(gradData, af::dim4(d0 + d1, d1, d2, d3));
    gradData = gradData.rows(0, d0 - 1);
    inputs[0].addGrad(fl::Variable(gradData, false));
  };
  return fl::Variable(data, {input}, gradFunc);
}

fl::Variable transformerMultiheadAttention(
    const fl::Variable& query,
    const fl::Variable& key,
    const fl::Variable& value,
    const fl::Variable& posEmb,
    const fl::Variable& mask,
    const int32_t nHead,
    const double pDropout,
    const int32_t offset = 0) {
  int32_t bsz = query.dims(2);
  int32_t modelDim = query.dims(1);
  int32_t headDim = modelDim / nHead;

  auto q = moddims(query, af::dim4(-1, headDim, nHead * bsz));
  auto k = moddims(key, af::dim4(-1, headDim, nHead * bsz));
  auto v = moddims(value, af::dim4(-1, headDim, nHead * bsz));

  auto scores = matmulNT(q, k);
  if (!posEmb.isempty()) {
    int n = posEmb.dims(0) / 2 - offset;
    auto pscores = transformerRotate(matmulNT(posEmb, q));
    scores = scores + transpose(pscores.rows(n, n + k.dims(0) - 1));
  }
  scores = scores / std::sqrt(float(headDim));
  if (!mask.isempty()) {
    scores = scores + tileAs(mask, scores);
  }

  auto attn = dropout(softmax(scores, 1), pDropout);
  auto result = matmul(attn, v);
  result = moddims(result, af::dim4(-1, headDim * nHead, bsz));
  return result;
}
} // namespace

namespace fl {

Transformer::Transformer(
    int32_t modelDim,
    int32_t headDim,
    int32_t mlpDim,
    int32_t nHeads,
    int32_t bptt,
    float pDropout,
    float pLayerdrop,
    bool useMask,
    bool preLN)
    : nHeads_(nHeads),
      bptt_(bptt),
      pDropout_(pDropout),
      pLayerdrop_(pLayerdrop),
      useMask_(useMask),
      preLN_(preLN),
      w1_(std::make_shared<Linear>(transformerInitLinear(modelDim, mlpDim))),
      w2_(std::make_shared<Linear>(transformerInitLinear(mlpDim, modelDim))),
      wq_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wk_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wv_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wf_(std::make_shared<Linear>(
          transformerInitLinear(headDim * nHeads, modelDim))),
      norm1_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}))),
      norm2_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}))) {
  if (bptt > 0) {
    params_.push_back(
        uniform(2 * bptt - 1, headDim, -0.1, 0.1, af::dtype::f32, true));
  }

  add(w1_);
  add(w2_);
  add(wq_);
  add(wk_);
  add(wv_);
  add(wf_);
  add(norm1_);
  add(norm2_);
}

Variable Transformer::mlp(const Variable& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  return (*w2_)(dropout(relu((*w1_)(input)), pDropout));
}

Variable Transformer::getMask(int32_t n, bool cache) {
  auto mask = af::lower(af::constant(1.0, n, n), true);
  if (cache) {
    auto maskCache = af::upper(af::constant(1.0, n, n));
    mask = af::join(1, maskCache, mask);
  }
  return Variable(af::log(mask), false);
}

Variable Transformer::selfAttention(const std::vector<Variable>& input) {
  int n = input[0].dims(1), bsz = input[0].dims(2);
  double pDrop = train_ ? pDropout_ : 0.0;

  auto q = transpose((*wq_)(input.back()));
  auto k = transpose((*wk_)(concatenate(input, 1)));
  auto v = transpose((*wv_)(concatenate(input, 1)));

  Variable mask, posEmb;
  if (bptt_ > 0) {
    posEmb = tile(params_[0], af::dim4(1, 1, nHeads_ * bsz));
  }
  if (useMask_ && input.back().dims(1) > 1) {
    mask = getMask(n, input.size() == 2);
  }

  int offset = (input.size() == 1) ? 0 : input[0].dims(1);

  auto result = transformerMultiheadAttention(
      q, k, v, posEmb, mask, nHeads_, pDrop, offset);
  result = (*wf_)(transpose(result));

  return result;
}

std::vector<Variable> Transformer::forward(const std::vector<Variable>& input) {
  auto x = input.back();
  float f = 1.0;
  if (train_ && (af::randu(1).scalar<float>() < pLayerdrop_)) {
    f = 0.0;
  }
  if (preLN_) {
    auto h = f * (*norm1_)(selfAttention(input)) + x;
    return {f * (*norm2_)(mlp(h)) + h};
  } else {
    auto h = (*norm1_)(f * selfAttention(input) + x);
    return {(*norm2_)(f * mlp(h) + h)};
  }
}

std::string Transformer::prettyString() const {
  return "Transformer";
}

Transformer::Transformer() {}

} // namespace fl
