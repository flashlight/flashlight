/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/Conformer.h"

#include <cmath>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

Conformer::Conformer(
    int32_t modelDim,
    int32_t headDim,
    int32_t mlpDim,
    int32_t nHeads,
    int32_t posEmbContextSize,
    int32_t convKernelSize,
    float pDropout,
    float pLayerDropout /* = 0. */)
    : nHeads_(nHeads),
      posEmbContextSize_(posEmbContextSize),
      convKernelSize_(convKernelSize),
      pDropout_(pDropout),
      pLayerDropout_(pLayerDropout),
      w11_(std::make_shared<Linear>(conformerInitLinear(modelDim, mlpDim))),
      w12_(std::make_shared<Linear>(conformerInitLinear(mlpDim, modelDim))),
      w21_(std::make_shared<Linear>(conformerInitLinear(modelDim, mlpDim))),
      w22_(std::make_shared<Linear>(conformerInitLinear(mlpDim, modelDim))),
      wq_(std::make_shared<Linear>(
          conformerInitLinear(modelDim, headDim * nHeads))),
      wk_(std::make_shared<Linear>(
          conformerInitLinear(modelDim, headDim * nHeads))),
      wv_(std::make_shared<Linear>(
          conformerInitLinear(modelDim, headDim * nHeads))),
      wf_(std::make_shared<Linear>(
          conformerInitLinear(headDim * nHeads, modelDim))),
      conv1_(std::make_shared<Linear>(
          conformerInitLinear(modelDim, modelDim * 2))),
      conv2_(std::make_shared<Linear>(conformerInitLinear(modelDim, modelDim))),
      norm1_(std::make_shared<LayerNorm>(
          std::vector<int>({0}),
          1e-5,
          true,
          modelDim)),
      norm2_(std::make_shared<LayerNorm>(
          std::vector<int>({0}),
          1e-5,
          true,
          modelDim)),
      normMhsa_(std::make_shared<LayerNorm>(
          std::vector<int>({0}),
          1e-5,
          true,
          modelDim)),
      normConv1_(std::make_shared<LayerNorm>(
          std::vector<int>({0}),
          1e-5,
          true,
          modelDim)),
      normConv2_(std::make_shared<LayerNorm>(
          std::vector<int>({0}),
          1e-5,
          true,
          modelDim)),
      norm3_(std::make_shared<LayerNorm>(
          std::vector<int>({0}),
          1e-5,
          true,
          modelDim)),
      convDepthWise_(std::make_shared<Conv2D>(
          modelDim,
          modelDim,
          convKernelSize,
          1,
          1,
          1,
          fl::PaddingMode::SAME,
          0,
          1,
          1,
          true,
          modelDim)) {
  if (posEmbContextSize_ > 0) {
    params_.push_back(uniform(2 * posEmbContextSize_ - 1, headDim, -0.1, 0.1));
  }
  createLayers();
}

Conformer::Conformer(const Conformer& other) {
  copy(other);
  createLayers();
}

Conformer& Conformer::operator=(const Conformer& other) {
  clear();
  copy(other);
  createLayers();
  return *this;
}

void Conformer::copy(const Conformer& other) {
  train_ = other.train_;
  nHeads_ = other.nHeads_;
  posEmbContextSize_ = other.posEmbContextSize_;
  convKernelSize_ = other.convKernelSize_;
  pDropout_ = other.pDropout_;
  pLayerDropout_ = other.pLayerDropout_;
  w11_ = std::make_shared<Linear>(*other.w11_);
  w12_ = std::make_shared<Linear>(*other.w12_);
  w21_ = std::make_shared<Linear>(*other.w21_);
  w22_ = std::make_shared<Linear>(*other.w22_);
  wq_ = std::make_shared<Linear>(*other.wq_);
  wk_ = std::make_shared<Linear>(*other.wk_);
  wv_ = std::make_shared<Linear>(*other.wv_);
  wf_ = std::make_shared<Linear>(*other.wf_);
  conv1_ = std::make_shared<Linear>(*other.conv1_);
  conv2_ = std::make_shared<Linear>(*other.conv2_);
  norm1_ = std::make_shared<LayerNorm>(*other.norm1_);
  norm2_ = std::make_shared<LayerNorm>(*other.norm2_);
  normMhsa_ = std::make_shared<LayerNorm>(*other.normMhsa_);
  normConv1_ = std::make_shared<LayerNorm>(*other.normConv1_);
  normConv2_ = std::make_shared<LayerNorm>(*other.normConv2_);
  norm3_ = std::make_shared<LayerNorm>(*other.norm3_);
  convDepthWise_ = std::make_shared<Conv2D>(*other.convDepthWise_);
  if (posEmbContextSize_ > 0) {
    const auto& p = other.param(0);
    params_.emplace_back(p.copy());
  }
}

void Conformer::createLayers() {
  // first feed-forward module
  add(w11_);
  add(w12_);
  add(norm1_);
  // second feed-forward module
  add(w21_);
  add(w22_);
  add(norm2_);
  // multihead attention module
  add(wq_);
  add(wk_);
  add(wv_);
  add(wf_);
  add(normMhsa_);
  // conv module
  add(conv1_);
  add(conv2_);
  add(convDepthWise_);
  add(normConv1_);
  add(normConv2_);
  // final layer norm of conformer block
  add(norm3_);
}

Variable Conformer::conformerInitLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std);
}

Variable Conformer::mhsa(const Variable& input, const Variable& inputPadMask) {
  float pDropout = train_ ? pDropout_ : 0.0;
  int bsz = input.dim(2);

  auto normedInput = (*normMhsa_)(input);
  auto q = transpose((*wq_)(normedInput), {1, 0, 2});
  auto k = transpose((*wk_)(normedInput), {1, 0, 2});
  auto v = transpose((*wv_)(normedInput), {1, 0, 2});

  Variable mask, posEmb;
  if (posEmbContextSize_ > 0) {
    posEmb = tile(params_[0].astype(input.type()), {1, 1, nHeads_ * bsz});
  }

  fl::Variable padMask;
  // TODO{fl::Tensor}{resize} - emulate the ArrayFire resize operation for
  // transformer pad mask
  if (!inputPadMask.isEmpty()) {
    auto padMaskArr = inputPadMask.tensor();
    Shape newMaskShape = {input.dim(1), input.dim(2)};
    if (padMaskArr.elements() != newMaskShape.elements()) {
      throw std::runtime_error(
          "Transformer::selfAttention - pad mask requires resize. "
          "This behavior will be fixed in a future release ");
    }
    padMaskArr = fl::reshape(padMaskArr, newMaskShape);
    padMask = fl::Variable(fl::log(padMaskArr), false);
  }

  auto result =
      multiheadAttention(q, k, v, posEmb, mask, padMask, nHeads_, pDropout, 0);
  result = (*wf_)(transpose(result, {1, 0, 2}));
  result = dropout(result, pDropout);
  return result;
}

Variable Conformer::conv(const Variable& _input) {
  // Make sure the input has 4 dims for depthwise conv
  Shape s = _input.shape();
  Variable input = moddims(_input, {s[0], s[1], s[2], 1});

  float pDropout = train_ ? pDropout_ : 0.0;
  // input C x T x B x 1
  // apply first pointwise conv
  auto result = gatedlinearunit(
      (*conv1_)(((*normConv1_)(input)).astype(input.type())), 0);
  result = reorder(result, {1, 3, 0, 2});
  // T x 1 x C x B
  // apply depthwise separable convolutions
  result = (*convDepthWise_)(result);
  result = reorder(result, {2, 0, 3, 1});
  // C x T x B x 1
  result = fl::swish(((*normConv2_)(result)).astype(input.type()), 1.);
  // apply second pointwise conv
  result = dropout((*conv2_)(result), pDropout);
  return moddims(result, _input.shape());
}

std::vector<Variable> Conformer::forward(const std::vector<Variable>& input) {
  if (input.size() != 2) {
    throw std::invalid_argument(
        "Invalid inputs for conformer block: there should be input "
        "and paddding mask (can be empty Variable)");
  }

  auto x = input[0];

  if (x.ndim() != 3) {
    throw std::invalid_argument(
        "Conformer::forward - input should be of 3 dimensions "
        "expects an input of size C x T x B - see documentation.");
  }

  float pDropout = train_ ? pDropout_ : 0.0;
  float f = 1.0;
  if (train_ && (fl::rand({1}).scalar<float>() < pLayerDropout_)) {
    f = 0.0;
  }
  // apply first feed-forward module
  auto ffn1 = dropout(
      (*w12_)(dropout(
          fl::swish((*w11_)(((*norm1_)(x)).astype(x.type())), 1.), pDropout)),
      pDropout);
  x = x + f * 0.5 * ffn1;
  // apply multihead attention module
  x = x + f * mhsa(x, input[1]);
  // apply conv module
  x = x + f * conv(x);
  // apply second feed-forward module
  auto ffn2 = dropout(
      (*w22_)(dropout(
          fl::swish((*w21_)(((*norm2_)(x)).astype(x.type())), 1.), pDropout)),
      pDropout);
  x = x + f * 0.5 * ffn2;
  x = ((*norm3_)(x)).astype(x.type());
  return {x};
}

std::unique_ptr<Module> Conformer::clone() const {
  return std::make_unique<Conformer>(*this);
}

std::string Conformer::prettyString() const {
  std::ostringstream ss;
  ss << "Conformer "
     << "(modelDim: " << params_[1].dim(1) << "), "
     << "(mlpDim: " << params_[1].dim(0) << "), "
     << "(nHeads: " << nHeads_ << "), "
     << "(pDropout: " << pDropout_ << "), "
     << "(pLayerDropout: " << pLayerDropout_ << "), "
     << "(posEmbContextSize: " << posEmbContextSize_ << "), "
     << "(convKernelSize: " << convKernelSize_ << ") ";
  return ss.str();
}

} // namespace fl
