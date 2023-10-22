/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/nn/VisionTransformer.h"

#include <cmath>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Random.h"

namespace fl::pkg::vision {

VisionTransformer::VisionTransformer(
    int32_t modelDim,
    int32_t headDim,
    int32_t mlpDim,
    int32_t nHeads,
    float pDropout,
    float pLayerdrop)
    : modelDim_(modelDim),
      headDim_(headDim),
      mlpDim_(mlpDim),
      nHeads_(nHeads),
      pDropout_(pDropout),
      pLayerdrop_(pLayerdrop),
      w1_(initLinear(modelDim, mlpDim)),
      w2_(initLinear(mlpDim, modelDim)),
      wq_(initLinear(modelDim, headDim * nHeads)),
      wk_(initLinear(modelDim, headDim * nHeads)),
      wv_(initLinear(modelDim, headDim * nHeads)),
      wf_(initLinear(headDim * nHeads, modelDim)),
      norm1_(std::make_shared<LayerNorm>(
          std::vector<int>({0}),
          1e-6, // eps
          true, // affine
          modelDim)),
      norm2_(std::make_shared<LayerNorm>(
          std::vector<int>({0}),
          1e-6, // eps
          true, // affine
          modelDim)) {
  createLayers();
}
VisionTransformer::VisionTransformer(const VisionTransformer& other) {
  copy(other);
  createLayers();
}
VisionTransformer& VisionTransformer::operator=(
    const VisionTransformer& other) {
  clear();
  copy(other);
  createLayers();
  return *this;
}

void VisionTransformer::copy(const VisionTransformer& other) {
  train_ = other.train_;
  modelDim_ = other.modelDim_;
  headDim_ = other.headDim_;
  mlpDim_ = other.mlpDim_;
  nHeads_ = other.nHeads_;
  pDropout_ = other.pDropout_;
  pLayerdrop_ = other.pLayerdrop_;
  w1_ = std::make_shared<Linear>(*other.w1_);
  w2_ = std::make_shared<Linear>(*other.w2_);
  wq_ = std::make_shared<Linear>(*other.wq_);
  wk_ = std::make_shared<Linear>(*other.wk_);
  wv_ = std::make_shared<Linear>(*other.wv_);
  wf_ = std::make_shared<Linear>(*other.wf_);
  norm1_ = std::make_shared<LayerNorm>(*other.norm1_);
  norm2_ = std::make_shared<LayerNorm>(*other.norm2_);
}

void VisionTransformer::createLayers() {
  add(w1_);
  add(w2_);
  add(wq_);
  add(wk_);
  add(wv_);
  add(wf_);
  add(norm1_);
  add(norm2_);
}

std::unique_ptr<Module> VisionTransformer::clone() const {
  throw std::runtime_error(
      "Cloning is unimplemented in Module 'VisionTransformer'");
}

Variable VisionTransformer::gelu(const Variable& input) {
  // https://arxiv.org/pdf/1606.08415.pdf
  auto geluConst = 1 / std::sqrt(2);
  auto res = 0.5 * input * (1 + erf(input * geluConst));
  return res;
}

Variable VisionTransformer::mlp(const Variable& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  auto output = (*w1_)(input);
  output = gelu(output.astype(fl::dtype::f32)).astype(input.type());
  output = dropout(output, pDropout);
  output = (*w2_)(output);
  output = dropout(output, pDropout);

  return output;
}

Variable VisionTransformer::selfAttention(const Variable& x) {
  // x - C x T x B
  double pDrop = train_ ? pDropout_ : 0.0;

  auto q = transpose((*wq_)(x), {1, 0, 2});
  auto k = transpose((*wk_)(x), {1, 0, 2});
  auto v = transpose((*wv_)(x), {1, 0, 2});

  auto result = multiheadAttention(
      q,
      k,
      v,
      fl::Variable(), // posEmb
      fl::Variable(), // mask
      fl::Variable(), // padMask
      nHeads_,
      pDrop,
      0 // offset
  );
  result = (*wf_)(transpose(result, {1, 0, 2}));
  result = dropout(result, pDrop);

  return result;
}

Variable VisionTransformer::dropPath(const Variable& x) {
  if (!train_) {
    return x;
  }

  // https://git.io/JYOkq
  int C = x.dim(0);
  int T = x.dim(1);
  int B = x.dim(2);
  auto keepMask = (fl::rand({1, 1, B}) > pLayerdrop_).astype(x.type());
  auto keepRatio =
      fl::mean(keepMask, {2}).astype(fl::dtype::f32).scalar<float>();
  // Note: this `keepRatio` is computed for real here, while in the PT
  // implementatino above, `keepRatio` = 1 - pLayerdrop_.
  keepMask = keepMask / keepRatio;
  return x * Variable(fl::tile(keepMask, {C, T}).astype(x.type()), false);
}

std::vector<Variable> VisionTransformer::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 1) {
    throw std::runtime_error("VisionTransformer forward, !1 input Variables");
  }

  auto x = inputs.front();

  if (x.ndim() != 3) {
    throw std::invalid_argument(
        "VisionTransformer::forward - "
        "expected input with 3 dimensions - got input with " +
        std::to_string(x.ndim()));
  }

  x = x + dropPath(selfAttention((*norm1_)(x)));
  x = x + dropPath(mlp((*norm2_)(x)));
  return {x};
}

std::string VisionTransformer::prettyString() const {
  std::ostringstream ss;
  ss << "VisionTransformer (nHeads: " << nHeads_ << "), "
     << "(modelDim_: " << modelDim_ << "), "
     << "(mlpDim_: " << mlpDim_ << "), "
     << "(pDropout: " << pDropout_ << "), "
     << "(pLayerdrop: " << pLayerdrop_ << "), ";
  return ss.str();
}

std::shared_ptr<fl::Linear> VisionTransformer::initLinear(
    int inDim,
    int outDim) {
  return std::make_shared<Linear>(
      fl::truncNormal({outDim, inDim}, 0.02),
      fl::constant(0., outDim, 1, fl::dtype::f32));
}
} // namespace fl
