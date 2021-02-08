/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/VisionTransformer.h"

#include <cmath>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace {

const float pi = std::acos(-1);
const float geluConst1 = std::sqrt(2 / pi);
const float geluConst2 = 0.044715;

} // namespace

namespace fl {

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
      w1_(std::make_shared<Linear>(
          initLinear(modelDim, mlpDim),
          fl::constant(0., mlpDim, 1))),
      w2_(std::make_shared<Linear>(
          initLinear(mlpDim, modelDim),
          fl::constant(0., modelDim, 1))),
      wq_(std::make_shared<Linear>(
          initLinear(modelDim, headDim * nHeads),
          fl::constant(0., headDim * nHeads, 1))),
      wk_(std::make_shared<Linear>(
          initLinear(modelDim, headDim * nHeads),
          fl::constant(0., headDim * nHeads, 1))),
      wv_(std::make_shared<Linear>(
          initLinear(modelDim, headDim * nHeads),
          fl::constant(0., headDim * nHeads, 1))),
      // wqkv_(std::make_shared<Linear>(
      //     initLinear(modelDim, headDim * nHeads * 3),
      //     fl::constant(0., headDim * nHeads * 3, 1))),
      wf_(std::make_shared<Linear>(
          initLinear(headDim * nHeads, modelDim),
          fl::constant(0., headDim * nHeads, 1))),
      norm1_(std::make_shared<LayerNorm>(
          std::vector<int>({0}),
          1e-6,
          true,
          modelDim)),
      norm2_(std::make_shared<LayerNorm>(
          std::vector<int>({0}),
          1e-6,
          true,
          modelDim)) {
  add(w1_);
  add(w2_);
  add(wq_);
  add(wk_);
  add(wv_);
  // add(wqkv_);
  add(wf_);
  add(norm1_);
  add(norm2_);
}

Variable VisionTransformer::gelu(const Variable& input) {
  // https://arxiv.org/pdf/1606.08415.pdf
  auto res = input + geluConst2 * pow(input, 3).as(input.type());
  res = 1. + tanh(geluConst1 * res).as(input.type());
  res = 0.5 * input * res;
  return res;
}

Variable VisionTransformer::mlp(const Variable& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  auto output = (*w1_)(input);
  output = gelu(output);
  output = dropout(output, pDropout);
  output = (*w2_)(output);
  output = dropout(output, pDropout);

  return output;
}

Variable VisionTransformer::selfAttention(const Variable& x) {
  // x - C x T x B
  auto T = x.dims(1);
  auto B = x.dims(2);

  double pDrop = train_ ? pDropout_ : 0.0;

  // 3 matrices
  auto q = transpose((*wq_)(x));
  auto k = transpose((*wk_)(x));
  auto v = transpose((*wv_)(x));

  // 1 matrix
  // auto qkv = (*wqkv_)(x);
  // qkv = moddims(qkv, af::dim4(modelDim_, 3, T, B));
  // qkv = reorder(qkv, 2, 0, 3, 1);
  // auto q = qkv(af::span, af::span, af::span, 0);
  // auto k = qkv(af::span, af::span, af::span, 1);
  // auto v = qkv(af::span, af::span, af::span, 2);

  q = moddims(q, af::dim4(-1, headDim_, nHeads_ * B));
  k = moddims(k, af::dim4(-1, headDim_, nHeads_ * B));
  v = moddims(v, af::dim4(-1, headDim_, nHeads_ * B));
  q = q / std::sqrt(float(headDim_));

  auto scores = matmulNT(q, k);
  auto attn = dropout(softmax(scores, 1), pDrop);
  auto result = matmul(attn.as(v.type()), v);
  result = moddims(result, af::dim4(-1, headDim_ * nHeads_, B));

  result = (*wf_)(transpose(result));
  result = dropout(result, pDrop);

  return result;
}

Variable VisionTransformer::dropPath(const Variable& x) {
  if (!train_) {
    return x;
  }

  // https://fburl.com/y59ssahb
  int C = x.dims(0);
  int T = x.dims(1);
  int B = x.dims(2);
  auto keepMask = (af::randu(1, 1, B) > pLayerdrop_).as(x.type());
  auto keepRatio = af::mean(keepMask, 2).as(f32).scalar<float>();
  keepMask = keepMask / keepRatio;
  // std::cout << keepRatio << std::endl;
  return x * Variable(af::tile(keepMask, af::dim4(C, T)).as(x.type()), false);
}

std::vector<Variable> VisionTransformer::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 1) {
    throw std::runtime_error("VisionTransformer forward, >1 inputs");
  }

  auto x = inputs.front();
  auto output = x + dropPath(selfAttention((*norm1_)(x)));
  output = output + dropPath(mlp((*norm2_)(output)));

  // double pLayerdrop = train_ ? pLayerdrop_ : 0.0;
  // float rand = 1. -
  //     std::floor(pLayerdrop +
  //                static_cast<float>(std::rand()) /
  //                    static_cast<float>(RAND_MAX));
  // auto output = x + rand * selfAttention((*norm1_)(x));
  // rand = 1. -
  //     std::floor(
  //            pLayerdrop +
  //            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
  // output = output + rand * mlp((*norm2_)(output));
  return {output};
}

std::string VisionTransformer::prettyString() const {
  std::ostringstream ss;
  ss << "VisionTransformer (nHeads: " << nHeads_ << "), "
     << "(pDropout: " << pDropout_ << "), "
     << "(pLayerdrop: " << pLayerdrop_ << "), ";
  return ss.str();
}

VisionTransformer::VisionTransformer() {}

fl::Variable VisionTransformer::initLinear(int32_t inDim, int32_t outDim) {
  // float std = std::sqrt(1.0 / float(inDim));
  // return fl::uniform(outDim, inDim, -std, std, af::dtype::f32, true);

  return truncNormal(af::dim4(outDim, inDim), 0.02);
}

} // namespace fl
