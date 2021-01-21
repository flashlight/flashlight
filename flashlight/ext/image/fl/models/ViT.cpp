/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/image/fl/models/ViT.h"

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
  // af::normal(af::dim4(hiddenEmbSize_, 1, 1, 1), 0.2), true);

  // Positional embedding
  // params_.emplace_back(
  //     VisionTransformer::initLinear(14 * 14 + 1, hiddenEmbSize_));
  params_.emplace_back(
      af::normal(af::dim4(hiddenEmbSize_, 14 * 14 + 1, 1, 1), 0.2), true);

  // Modules
  add(patchEmbedding_);

  for (int i = 0; i < nLayers_; ++i) {
    // transformers_.emplace_back(std::make_shared<Transformer>(
    //     hiddenEmbSize_,
    //     hiddenEmbSize_ / nHeads_,
    //     mlpSize_,
    //     nHeads_,
    //     0,
    //     pDropout,
    //     pLayerDrop));
    transformers_.emplace_back(std::make_shared<VisionTransformer>(
        hiddenEmbSize_,
        hiddenEmbSize_ / nHeads_,
        mlpSize_,
        nHeads_,
        pDropout,
        pLayerDrop));
    add(transformers_.back());
  }

  linearOut_ = std::make_shared<Linear>(
      VisionTransformer::initLinear(hiddenEmbSize_, nClasses));
  add(linearOut_);

  ln_ = std::make_shared<LayerNorm>(std::vector<int>({0, 3}), 1e-6);
  add(ln_);
}

std::vector<fl::Variable> ViT::forward(
    const std::vector<fl::Variable>& inputs) {
  // Patching
  auto output = patchEmbedding_->forward(inputs[0]); // H x W x C x B
  output = moddims(output, af::dim4(-1, 1, 0, 0)); // T x 1 x C x B
  output = reorder(output, 2, 0, 3, 1); // C x T x B
  auto B = output.dims(2);

  // Prepending the class token
  auto clsToken = tile(params_[0], af::dim4(1, 1, B)); // C x 1 x B
  output = concatenate({clsToken, output}, 1);

  // Positional embedding
  auto posEmb = tile(params_[1], af::dim4(1, 1, B));
  output = output + posEmb;
  // if (train_) {
  //   output = dropout(output, pDropout_);
  // }

  // Transformers
  for (int i = 0; i < nLayers_; ++i) {
    output = transformers_[i]->forward({output}).front();
    // output =
    //     transformers_[i]->forward({output, fl::noGrad(af::array())}).front();
  }

  // Linear
  output = ln_->forward(output);
  output = reorder(output, 0, 2, 1, 3).slice(0); // C x B x 1
  output = linearOut_->forward(output);

  return {logSoftmax(output, 0)};
}

std::string ViT::prettyString() const {
  std::ostringstream ss;
  ss << "ViT with " << nLayers_ << " Transformers:\n"
     << transformers_.front()->prettyString();
  return ss.str();
}

} // namespace image
} // namespace ext
} // namespace fl