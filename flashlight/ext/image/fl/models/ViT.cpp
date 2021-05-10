/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/image/fl/models/ViT.h"

#include <fstream>

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
    const int nClasses,
    const bool usePosEmb,
    const bool useAugPosEmb,
    const int imageSize)
    : nLayers_(nLayers),
      hiddenEmbSize_(hiddenEmbSize),
      mlpSize_(mlpSize),
      nHeads_(nHeads),
      pDropout_(pDropout),
      nClasses_(nClasses),
      usePosEmb_(usePosEmb),
      useAugPosEmb_(useAugPosEmb),
      patchEmbedding_(
          std::make_shared<Conv2D>(3, hiddenEmbSize_, 16, 16, 16, 16)) {
  // Class token
  params_.emplace_back(fl::truncNormal(af::dim4(hiddenEmbSize_, 1), 0.02));

  // Positional embedding
  if (usePosEmb_) {
    int nPatches = imageSize / 16;
    params_.emplace_back(
        fl::truncNormal(af::dim4(hiddenEmbSize_, nPatches * nPatches + 1), 0.02));
  } else if (useAugPosEmb_) {
    augPosEmb_ =
        std::make_shared<SinusoidalPositionEmbedding2D>(hiddenEmbSize_, 1.);
    add(augPosEmb_);
  }

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
      fl::truncNormal(af::dim4(nClasses, hiddenEmbSize_), 0.02),
      fl::constant(0., nClasses, 1, af::dtype::f32));
  add(linearOut_);

  ln_ = std::make_shared<LayerNorm>(
      std::vector<int>({0}), 1e-6, true, hiddenEmbSize_);
  add(ln_);
}

std::vector<fl::Variable> ViT::forward(
    const std::vector<fl::Variable>& inputs) {
  return forward(inputs, false);
}

std::vector<fl::Variable> ViT::forward(
    const std::vector<fl::Variable>& inputs,
    bool useFp16,
    bool eval,
    bool aug,
    float region,
    bool doShrink) {
  // Patching
  auto output = patchEmbedding_->forward(inputs[0]); // H x W x C x B
  output = moddims(output, af::dim4(-1, 1, 0, 0)); // T x 1 x C x B
  output = reorder(output, 2, 0, 3, 1); // C x T x B
  auto B = output.dims(2);

  // Prepending the class token
  auto clsToken =
      tile(params_[0], af::dim4(1, 1, B)).as(output.type()); // C x 1 x B
  output = concatenate({clsToken, output}, 1);

  // Positional embedding
  if (usePosEmb_) {
    auto posEmb = params_[1];
    if (eval && posEmb.dims(1) != output.dims(1)) {
      af::array posEmbArr = params_[1].array();
      af::array clsEmb = posEmbArr.col(0);
      af::array posEmbAll = posEmbArr.cols(1, posEmbArr.dims(1) - 1);
      auto newPosEmb = af::resize(posEmbAll, output.dims(0), output.dims(1) - 1, AF_INTERP_BICUBIC);
      posEmb = fl::Variable(af::join(1, clsEmb, newPosEmb), false);
    }
    output = output + tile(posEmb, af::dim4(1, 1, B)).as(output.type());;
  } else if (useAugPosEmb_) {
    output = augPosEmb_->forward({output}, aug, region, doShrink).front();
  }
  if (train_) {
    output = dropout(output, pDropout_);
  }

  if (useFp16) {
    // Avoid running conv2D in fp16 in this case.
    // All the other part of the network can be run in fp16 if compatible.
    output = output.as(f16);
  }
  // Transformers
  for (int i = 0; i < nLayers_; ++i) {
    output = transformers_[i]->forward({output}).front();
  }
  // Linear
  output = ln_->forward(output); // C x T x B
  output = reorder(output, 0, 2, 1, 3).slice(0); // C x B x 1
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

} // namespace image
} // namespace ext
} // namespace fl
