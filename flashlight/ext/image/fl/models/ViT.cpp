/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/image/fl/models/ViT.h"

#include <fstream>
#include <iostream>

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

namespace fl {
namespace ext {
namespace image {

ViT::ViT(
    const int nLayers,
    const int hiddenEmbSize,
    const int mlpSize,
    const int nHeads,
    const float pDropout,
    const float pLayerDrop,
    const int nClasses)
    : nLayers_(nLayers),
      hiddenEmbSize_(hiddenEmbSize),
      mlpSize_(mlpSize),
      nHeads_(nHeads),
      pDropout_(pDropout),
      patchEmbedding_(
          std::make_shared<Conv2D>(3, hiddenEmbSize_, 16, 16, 16, 16)) {
  // Class token
  params_.emplace_back(VisionTransformer::initLinear(1, hiddenEmbSize_));
  params_.back().disableWeightDecay();

  // Positional embedding
  params_.emplace_back(
      VisionTransformer::initLinear(14 * 14 + 1, hiddenEmbSize_));
  params_.back().disableWeightDecay();

  // Modules
  add(patchEmbedding_);

  for (int i = 0; i < nLayers_; ++i) {
    transformers_.emplace_back(std::make_shared<VisionTransformer>(
        hiddenEmbSize_,
        hiddenEmbSize_ / nHeads_,
        mlpSize_,
        nHeads_,
        pDropout,
        pLayerDrop * i / (nLayers_ - 1)));
    add(transformers_.back());
  }

  linearOut_ = std::make_shared<Linear>(
      VisionTransformer::initLinear(hiddenEmbSize_, nClasses),
      fl::constant(0., nClasses, 1, af::dtype::f32));
  add(linearOut_);

  ln_ = std::make_shared<LayerNorm>(
      std::vector<int>({0}), 1e-6, true, hiddenEmbSize_);
  add(ln_);
}

std::vector<fl::Variable> ViT::forward(
    const std::vector<fl::Variable>& inputs) {
  // std::cout << inputs[0].type() << " ";

  // Patching
  auto output = patchEmbedding_->forward(inputs[0]); // H x W x C x B
  output = output.as(f16);
  output = moddims(output, af::dim4(-1, 1, 0, 0)); // T x 1 x C x B
  output = reorder(output, 2, 0, 3, 1); // C x T x B
  auto B = output.dims(2);
  // std::cout << output.type() << " ";

  // Prepending the class token
  auto clsToken =
      tile(params_[0], af::dim4(1, 1, B)).as(output.type()); // C x 1 x B
  output = concatenate({clsToken, output}, 1);
  // std::cout << output.type() << " ";

  // Positional embedding
  auto posEmb = tile(params_[1], af::dim4(1, 1, B)).as(output.type());
  output = output + posEmb;
  if (train_) {
    output = dropout(output, pDropout_);
  }
  // std::cout << output.type() << " ";
  // std::cout << " - 0 - \n";
  // af_print(output.array()(af::seq(0, 9), 0, 0));

  // Transformers
  for (int i = 0; i < nLayers_; ++i) {
    output = transformers_[i]->forward({output}).front();
    // std::cout << " - " << i << " - \n";
    // af_print(output.array()(af::seq(0, 9), 0, 0));
  }
  // std::cout << output.type() << " ";

  // Linear
  output = ln_->forward(output); // C x T x B
  // std::cout << output.type() << " ";
  output = reorder(output, 0, 2, 1, 3).slice(0); // C x B x 1
  // std::cout << output.type() << " ";
  output = linearOut_->forward(output);
  // std::cout << output.type() << " ";
  output = logSoftmax(output, 0).as(output.type());
  // std::cout << output.type() << std::endl;

  return {output};
}

std::string ViT::prettyString() const {
  std::ostringstream ss;
  ss << "ViT with " << nLayers_ << " Transformers:\n";
  for (const auto& transformers : transformers_) {
    ss << transformers->prettyString() << "\n";
  }
  return ss.str();
}

ViT::ViT(const std::string& prefix)
    : nLayers_(12),
      hiddenEmbSize_(768),
      mlpSize_(3072),
      nHeads_(12),
      pDropout_(0.) {
  // Class token
  auto w = readfloats(prefix + "cls_token.bin");
  if (w.size() != hiddenEmbSize_) {
    throw std::runtime_error("invalid cls");
  }
  params_.emplace_back(fl::Variable(af::array(hiddenEmbSize_, w.data())));
  params_.back().disableWeightDecay();

  // Positional embedding
  w = readfloats(prefix + "pos_embed.bin");
  if (w.size() != hiddenEmbSize_ * 197) {
    throw std::runtime_error("invalid pos_embed.bin");
  }
  params_.emplace_back(
      fl::Variable(af::array(hiddenEmbSize_, 14 * 14 + 1, w.data())));
  // af_print(params_.back().array()(af::span, 0));
  params_.back().disableWeightDecay();

  // Modules
  w = readfloats(prefix + "patch_embed.proj.weight.bin");
  if (w.size() != 768 * 3 * 16 * 16) {
    throw std::runtime_error("invalid patch_embed.proj.weight.bin");
  }
  auto b = readfloats(prefix + "patch_embed.proj.bias.bin");
  if (b.size() != 768) {
    throw std::runtime_error("invalid patch_embed.proj.bias.bin");
  }
  auto arr_w = fl::Variable(af::array(16, 16, 3, 768, w.data()));
  auto arr_b = fl::Variable(af::array(1, 1, 768, 1, b.data()));
  // af_print(arr_w.array()(af::span, 0, 0, 0));
  patchEmbedding_ = std::make_shared<Conv2D>(arr_w, arr_b, 16, 16);
  add(patchEmbedding_);

  for (int i = 0; i < nLayers_; ++i) {
    transformers_.emplace_back(std::make_shared<VisionTransformer>(
        prefix + "blocks." + std::to_string(i), 0.1 * i / (nLayers_ - 1)));
    add(transformers_.back());
  }

  w = readfloats(prefix + "norm.weight.bin");
  if (w.size() != 768) {
    throw std::runtime_error("norm.weight.bin");
  }
  b = readfloats(prefix + "norm.bias.bin");
  if (b.size() != 768) {
    throw std::runtime_error("norm.bias.bin");
  }
  arr_w = fl::Variable(af::array(768, w.data()));
  arr_b = fl::Variable(af::array(768, b.data()));
  ln_ = std::make_shared<LayerNorm>(
      std::vector<int>({0}), 1e-6, true, hiddenEmbSize_);
  ln_->setParams(arr_w, 0);
  ln_->setParams(arr_b, 1);
  add(ln_);

  w = readfloats(prefix + "head.weight.bin");
  if (w.size() != 768 * 1000) {
    throw std::runtime_error("head.weight.bin");
  }
  b = readfloats(prefix + "head.bias.bin");
  if (b.size() != 1000) {
    throw std::runtime_error("head.bias.bin");
  }
  arr_w = fl::Variable(af::array(1000, 768, w.data()));
  arr_b = fl::Variable(af::array(1000, b.data()));
  // af_print(arr_w.array()(af::span, 0));
  linearOut_ = std::make_shared<Linear>(arr_w, arr_b);
  add(linearOut_);
}

} // namespace image
} // namespace ext
} // namespace fl