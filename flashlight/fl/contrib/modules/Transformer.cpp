/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/contrib/modules/Transformer.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"
#include "flashlight/fl/tensor/Random.h"

namespace {
fl::Variable transformerInitLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std, fl::dtype::f32, true);
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
        uniform(2 * bptt - 1, headDim, -0.1, 0.1, fl::dtype::f32, true));
  }

  createLayers();
}

Transformer::Transformer(const Transformer& other) {
  copy(other);
  createLayers();
}

Transformer& Transformer::operator=(const Transformer& other) {
  clear();
  copy(other);
  createLayers();
  return *this;
}

void Transformer::copy(const Transformer& other) {
  train_ = other.train_;
  nHeads_ = other.nHeads_;
  bptt_ = other.bptt_;
  pDropout_ = other.pDropout_;
  pLayerdrop_ = other.pLayerdrop_;
  useMask_ = other.useMask_;
  preLN_ = other.preLN_;
  w1_ = std::make_shared<Linear>(*other.w1_);
  w2_ = std::make_shared<Linear>(*other.w2_);
  wq_ = std::make_shared<Linear>(*other.wq_);
  wk_ = std::make_shared<Linear>(*other.wk_);
  wv_ = std::make_shared<Linear>(*other.wv_);
  wf_ = std::make_shared<Linear>(*other.wf_);
  norm1_ = std::make_shared<LayerNorm>(*other.norm1_);
  norm2_ = std::make_shared<LayerNorm>(*other.norm2_);
  if (bptt_ > 0) {
    const auto& p = other.param(0);
    params_.emplace_back(p.copy());
  }
}

void Transformer::createLayers() {
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
  auto mask = fl::tril(fl::full({n, n}, 1.0));
  if (cache) {
    auto maskCache = fl::triu(fl::full({n, n}, 1.0));
    mask = fl::concatenate(1, maskCache, mask);
  }
  return Variable(fl::log(mask), false);
}

Variable Transformer::selfAttention(const std::vector<Variable>& input) {
  // previous step[optionally], input, padMask
  const auto& encoderInput = input.at(input.size() - 2);
  // in case of previous state input[0] has size CxT_prevxB
  int n = input[0].dim(1), bsz = input[0].dim(2);
  double pDrop = train_ ? pDropout_ : 0.0;

  auto q = transpose((*wq_)(encoderInput), {1, 0, 2});
  std::vector<fl::Variable> inputWithState(input.begin(), input.end() - 1);
  auto k = transpose((*wk_)(concatenate(inputWithState, 1)), {1, 0, 2});
  auto v = transpose((*wv_)(concatenate(inputWithState, 1)), {1, 0, 2});

  Variable mask, posEmb;
  if (bptt_ > 0) {
    posEmb =
        tile(params_[0].astype(encoderInput.type()), {1, 1, nHeads_ * bsz});
  }
  if (useMask_ && encoderInput.dim(1) > 1) {
    // mask future if we use the previous state (then n is previous time)
    mask = getMask(n, input.size() == 3);
  }

  int offset = (input.size() == 2) ? 0 : n;

  // time x batch
  fl::Variable padMask;
  if (!input.back().isEmpty()) {
    auto padMaskArr = input.back().tensor();
    Shape newMaskShape = {encoderInput.dim(1), encoderInput.dim(2)};
    // TODO{fl::Tensor}{resize} - emulate the ArrayFire resize operation for
    // transformer pad mask
    if (padMaskArr.elements() != newMaskShape.elements()) {
      throw std::runtime_error(
          "Transformer::selfAttention - pad mask requires resize. "
          "This behavior will be fixed in a future release ");
    }
    padMaskArr = fl::reshape(padMaskArr, newMaskShape);
    padMask = fl::Variable(fl::log(padMaskArr), false);
  }
  auto result = multiheadAttention(
      q, k, v, posEmb, mask, padMask, nHeads_, pDrop, offset);
  result = (*wf_)(transpose(result, {1, 0, 2}));

  return result;
}

std::vector<Variable> Transformer::forward(const std::vector<Variable>& input) {
  // previous step[optionally], input, padMask
  // padMask should be empty if previous step is provided
  // padMask is expected to have "1" on the used positions and "0" on padded
  // positions
  if (input.size() != 2) {
    throw std::invalid_argument(
        "Invalid inputs for transformer block: there should be at least input and mask");
  }
  const auto& x = input.at(input.size() - 2);
  if (x.ndim() != 3) {
    throw std::invalid_argument(
        "Transformer::forward - input should be of 3 dimensions "
        "expects an input of size C x T x B - see documentation.");
  }

  if (!input.back().isEmpty()) {
    if (input.back().ndim() < 2) {
      throw std::invalid_argument(
          "Transformer::forward - invalid size for pad mask - "
          "must have at least two dimensions");

    } else if (x.dim(2) != input.back().dim(1)) {
      throw std::invalid_argument(
          "Transformer::forward - invalid inputs for transformer:"
          " input and mask batch sizes are different");
    }
  }

  float f = 1.0;
  if (train_ && (fl::rand({1}).scalar<float>() < pLayerdrop_)) {
    f = 0.0;
  }
  if (preLN_) {
    auto h = (f * (*norm1_)(selfAttention(input))).astype(x.type()) + x;
    return {f * (*norm2_)(mlp(h)).astype(h.type()) + h};
  } else {
    auto h = (*norm1_)((f * selfAttention(input)).astype(x.type()) + x);
    return {(*norm2_)((f * mlp(h)).astype(h.type()) + h)};
  }
}

void Transformer::setDropout(float value) {
  pDropout_ = value;
}

void Transformer::setLayerDropout(float value) {
  pLayerdrop_ = value;
}

std::unique_ptr<Module> Transformer::clone() const {
  return std::make_unique<Transformer>(*this);
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

Transformer::Transformer() = default;

} // namespace fl
