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

#include <fstream>
#include <iostream>

namespace {

const float geluConst = 1 / std::sqrt(2);

std::vector<float> readfloats(const std::string& filepath) {
  std::ifstream fin(filepath, std::ios::binary);
  if (!fin) {
    std::cout << " Error, Couldn't find the file\n";
    return {};
  }

  fin.seekg(0, std::ios::end);
  const size_t num_elements = fin.tellg() / sizeof(float);
  fin.seekg(0, std::ios::beg);

  std::vector<float> res(num_elements);
  fin.read(reinterpret_cast<char*>(res.data()), num_elements * sizeof(float));
  return res;
}

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
          fl::constant(0., mlpDim, 1, af::dtype::f32))),
      w2_(std::make_shared<Linear>(
          initLinear(mlpDim, modelDim),
          fl::constant(0., modelDim, 1, af::dtype::f32))),
      wq_(std::make_shared<Linear>(
          initLinear(modelDim, headDim * nHeads),
          fl::constant(0., headDim * nHeads, 1, af::dtype::f32))),
      wk_(std::make_shared<Linear>(
          initLinear(modelDim, headDim * nHeads),
          fl::constant(0., headDim * nHeads, 1, af::dtype::f32))),
      wv_(std::make_shared<Linear>(
          initLinear(modelDim, headDim * nHeads),
          fl::constant(0., headDim * nHeads, 1, af::dtype::f32))),
      // wqkv_(std::make_shared<Linear>(
      //     initLinear(modelDim, headDim * nHeads * 3),
      //     fl::constant(0., headDim * nHeads * 3, 1))),
      wf_(std::make_shared<Linear>(
          initLinear(headDim * nHeads, modelDim),
          fl::constant(0., headDim * nHeads, 1, af::dtype::f32))),
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
  auto res = 0.5 * input * (1 + erf(input * geluConst));
  return res;
}

Variable VisionTransformer::mlp(const Variable& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  auto output = (*w1_)(input);
  // std::cout << " ++ " << 1 << " - \n";
  // af_print(output.array()(af::seq(0, 9), 0, 0));
  output = gelu(output.as(f32)).as(input.type());
  // std::cout << " ++ " << 2 << " - \n";
  // af_print(output.array()(af::seq(0, 9), 0, 0));
  output = dropout(output, pDropout);
  output = (*w2_)(output);
  // std::cout << " ++ " << 3 << " - \n";
  // af_print(output.array()(af::seq(0, 9), 0, 0));
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
  auto attn = softmax(scores, 1);
  // attn = dropout(attn, pDrop);
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
  // std::cout << " + " << 1 << " - \n";
  // af_print(x.array()(af::seq(0, 9), 0, 0));
  x = x + dropPath(selfAttention((*norm1_)(x)));
  // std::cout << " + " << 2 << " - \n";
  // af_print(x.array()(af::seq(0, 9), 0, 0));
  x = x + dropPath(mlp((*norm2_)(x)));
  return {x};
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

VisionTransformer::VisionTransformer(
    const std::string& prefix,
    float pLayerdrop)
    : modelDim_(768),
      headDim_(768 / 12),
      mlpDim_(768 * 4),
      nHeads_(12),
      pDropout_(0.),
      pLayerdrop_(pLayerdrop) {
  auto w = readfloats(prefix + ".mlp.fc1.weight.bin");
  if (w.size() != 768 * 3072) {
    throw std::runtime_error(".mlp.fc1.weight.bin");
  }
  auto b = readfloats(prefix + ".mlp.fc1.bias.bin");
  if (b.size() != 3072) {
    throw std::runtime_error(".mlp.fc1.bias.bin");
  }
  auto arr_w = fl::Variable(af::array(3072, 768, w.data()));
  auto arr_b = fl::Variable(af::array(3072, b.data()));
  w1_ = std::make_shared<Linear>(arr_w, arr_b);

  w = readfloats(prefix + ".mlp.fc2.weight.bin");
  if (w.size() != 768 * 3072) {
    throw std::runtime_error(".mlp.fc2.weight.bin");
  }
  b = readfloats(prefix + ".mlp.fc2.bias.bin");
  if (b.size() != 768) {
    throw std::runtime_error(".mlp.fc2.bias.bin");
  }
  arr_w = fl::Variable(af::array(768, 3072, w.data()));
  arr_b = fl::Variable(af::array(768, b.data()));
  w2_ = std::make_shared<Linear>(arr_w, arr_b);

  w = readfloats(prefix + ".attn.qkv.weight.bin");
  if (w.size() != 768 * 2304) {
    throw std::runtime_error(".attn.qkv.weight.bin");
  }
  b = readfloats(prefix + ".attn.qkv.bias.bin");
  if (b.size() != 2304) {
    throw std::runtime_error(".attn.qkv.bias.bin");
  }
  auto all_w = af::array(2304, 768, w.data());
  auto all_b = af::array(2304, b.data());

  arr_w = fl::Variable(all_w(af::seq(0, 768 - 1), af::span));
  arr_b = fl::Variable(all_b(af::seq(0, 768 - 1)));
  arr_w.linear();
  arr_b.linear();
  wq_ = std::make_shared<Linear>(arr_w, arr_b);

  arr_w = fl::Variable(all_w(af::seq(768, 2 * 768 - 1), af::span));
  arr_b = fl::Variable(all_b(af::seq(768, 2 * 768 - 1)));
  arr_w.linear();
  arr_b.linear();
  wk_ = std::make_shared<Linear>(arr_w, arr_b);

  arr_w = fl::Variable(all_w(af::seq(2 * 768, 3 * 768 - 1), af::span));
  arr_b = fl::Variable(all_b(af::seq(2 * 768, 3 * 768 - 1)));
  arr_w.linear();
  arr_b.linear();
  wv_ = std::make_shared<Linear>(arr_w, arr_b);

  w = readfloats(prefix + ".attn.proj.weight.bin");
  if (w.size() != 768 * 768) {
    throw std::runtime_error(".attn.proj.weight.bin");
  }
  b = readfloats(prefix + ".attn.proj.bias.bin");
  if (b.size() != 768) {
    throw std::runtime_error(".attn.proj.bias.bin");
  }
  arr_w = fl::Variable(af::array(768, 768, w.data()));
  arr_b = fl::Variable(af::array(768, b.data()));
  wf_ = std::make_shared<Linear>(arr_w, arr_b);

  w = readfloats(prefix + ".norm1.weight.bin");
  if (w.size() != 768) {
    throw std::runtime_error(".norm1.weight.bin");
  }
  b = readfloats(prefix + ".norm1.bias.bin");
  if (b.size() != 768) {
    throw std::runtime_error(".norm1.bias.bin");
  }
  arr_w = fl::Variable(af::array(768, w.data()));
  arr_b = fl::Variable(af::array(768, b.data()));
  norm1_ =
      std::make_shared<LayerNorm>(std::vector<int>({0}), 1e-6, true, modelDim_);
  norm1_->setParams(arr_w, 0);
  norm1_->setParams(arr_b, 1);

  w = readfloats(prefix + ".norm2.weight.bin");
  if (w.size() != 768) {
    throw std::runtime_error(".norm2.weight.bin");
  }
  b = readfloats(prefix + ".norm2.bias.bin");
  if (b.size() != 768) {
    throw std::runtime_error(".norm2.bias.bin");
  }
  arr_w = fl::Variable(af::array(768, w.data()));
  arr_b = fl::Variable(af::array(768, b.data()));
  norm2_ =
      std::make_shared<LayerNorm>(std::vector<int>({0}), 1e-6, true, modelDim_);
  norm2_->setParams(arr_w, 0);
  norm2_->setParams(arr_b, 1);

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

} // namespace fl
