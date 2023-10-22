/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/models/ViT.h"

#include <sstream>

#include "flashlight/fl/tensor/Index.h"

namespace fl::pkg::vision {

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
      nClasses_(nClasses),
      patchEmbedding_(
          std::make_shared<Conv2D>(3, hiddenEmbSize_, 16, 16, 16, 16)) {
  // Class token
  params_.emplace_back(fl::truncNormal({hiddenEmbSize_, 1}, 0.02));

  // Positional embedding
  params_.emplace_back(fl::truncNormal({hiddenEmbSize_, 14 * 14 + 1}, 0.02));

  // Modules
  add(patchEmbedding_);

  for (int i = 0; i < nLayers_; ++i) {
    transformers_.emplace_back(std::make_shared<VisionTransformer>(
        hiddenEmbSize_,
        hiddenEmbSize_ / nHeads_,
        mlpSize_,
        nHeads_,
        pDropout_,
        pLayerDrop * i / (nLayers_ - 1)));
    add(transformers_.back());
  }

  linearOut_ = std::make_shared<Linear>(
      fl::truncNormal({nClasses_, hiddenEmbSize_}, 0.02),
      fl::constant(0., nClasses_, 1, fl::dtype::f32));
  add(linearOut_);

  ln_ = std::make_shared<LayerNorm>(
      std::vector<int>({0}), 1e-6, true, hiddenEmbSize_);
  add(ln_);
}

void ViT::copy(const ViT& other) {
  clear();

  // Class token
  auto clsTkn = other.param(0);
  params_.emplace_back(clsTkn.copy());

  // Positional embedding
  auto posEmb = other.param(1);
  params_.emplace_back(posEmb.copy());

  // Modules
  patchEmbedding_ = std::make_shared<Conv2D>(*other.patchEmbedding_);
  add(patchEmbedding_);

  for (const auto& vit : other.transformers_) {
    transformers_.emplace_back(std::make_shared<VisionTransformer>(*vit));
    add(transformers_.back());
  }

  linearOut_ = std::make_shared<Linear>(*other.linearOut_);
  add(linearOut_);

  ln_ = std::make_shared<LayerNorm>(*other.ln_);
  add(ln_);
}

ViT::ViT(const ViT& other) {
  copy(other);
}

ViT& ViT::operator=(const ViT& other) {
  copy(other);
  return *this;
}

std::unique_ptr<Module> ViT::clone() const {
  return std::make_unique<ViT>(*this);
}

std::vector<fl::Variable> ViT::forward(
    const std::vector<fl::Variable>& inputs) {
  // Patching
  auto output = patchEmbedding_->forward(inputs[0]); // H x W x C x B
  output = moddims(output, {-1, 1, 0, 0}); // T x 1 x C x B
  output = reorder(output, {2, 0, 3, 1}); // C x T x B x 1
  output = moddims(output, {0, 0, 0}); // C x T x B
  auto B = output.dim(2);

  // Prepending the class token
  auto clsToken =
      tile(params_[0], {1, 1, B}).astype(output.type()); // C x 1 x B
  output = concatenate({clsToken, output}, 1);

  // Positional embedding
  auto posEmb = tile(params_[1], {1, 1, B}).astype(output.type());
  output = output + posEmb;
  if (train_) {
    output = dropout(output, pDropout_);
  }

  // Transformers
  for (int i = 0; i < nLayers_; ++i) {
    output = transformers_[i]->forward({output}).front();
  }

  // Linear
  output = ln_->forward(output); // C x T x B
  output = reorder(output, {0, 2, 1})(fl::span, fl::span, 0); // C x B x 1
  output = linearOut_->forward(output);

  return {output};
}

std::string ViT::prettyString() const {
  std::ostringstream ss;
  ss << "ViT (" << nClasses_ << " classes) with " << nLayers_
     << " Transformers:\n";
  for (const auto& transformers : transformers_) {
    ss << transformers->prettyString() << "\n";
  }
  return ss.str();
}

} // namespace fl
