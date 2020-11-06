/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/Transformer.h"
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace {
fl::Variable transformerInitLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std, af::dtype::f32, true);
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
    posEmb =
        tile(params_[0].as(input.back().type()), af::dim4(1, 1, nHeads_ * bsz));
  }
  if (useMask_ && input.back().dims(1) > 1) {
    mask = getMask(n, input.size() == 2);
  }

  int offset = (input.size() == 1) ? 0 : n;

  auto result = multiheadAttention(q, k, v, posEmb, mask, nHeads_, pDrop, offset);
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
    auto h = (f * (*norm1_)(selfAttention(input))).as(x.type()) + x;
    return {f * (*norm2_)(mlp(h)).as(h.type()) + h};
  } else {
    auto h = (*norm1_)((f * selfAttention(input)).as(x.type()) + x);
    return {(*norm2_)((f * mlp(h)).as(h.type()) + h)};
  }
}

std::string Transformer::prettyString() const {
  std::ostringstream ss;
  ss << "Transformer (nHeads: " << nHeads_ << "), "
     << "(pDropout: " << pDropout_ << "), "
     << "(pLayerdrop: " << pLayerdrop_ << "), "
     << "(bptt: " << bptt_ << "), "
     << "(useMask: " << useMask_ << "), "
     << "(preLayerNorm: " << preLN_ << ")";
  return ss.str();
}

Transformer::Transformer() {}

} // namespace fl
