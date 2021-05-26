/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/vision/nn/VisionTransformer.h"
#include "flashlight/fl/nn/nn.h"

namespace fl {
namespace pkg {
namespace vision {

/*
 * Implementation of Vision Transformer (ViT) models following [AN IMAGE IS
 * WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT
 * SCALE](https://arxiv.org/pdf/2010.11929.pdf)
 *
 * This implementation is highly inspired by [timm](https://git.io/JYOql).
 */
class ViT : public fl::Container {
 private:
  FL_SAVE_LOAD_WITH_BASE(
      fl::Container,
      nLayers_,
      hiddenEmbSize_,
      mlpSize_,
      nHeads_,
      pDropout_,
      patchEmbedding_,
      transformers_,
      linearOut_,
      ln_)

  int nLayers_;
  int hiddenEmbSize_;
  int mlpSize_;
  int nHeads_;
  float pDropout_;
  int nClasses_;

  std::shared_ptr<Conv2D> patchEmbedding_;
  std::vector<std::shared_ptr<VisionTransformer>> transformers_;
  std::shared_ptr<Linear> linearOut_;
  std::shared_ptr<LayerNorm> ln_;

  ViT() = default;

 public:
  ViT(const int nLayers,
      const int hiddenEmbSize,
      const int mlpSize,
      const int nHeads,
      const float pDropout,
      const float pLayerDrop,
      const int nClasses);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  std::string prettyString() const override;
};

} // namespace vision
} // namespace pkg
} // namespace fl

CEREAL_REGISTER_TYPE(fl::pkg::vision::ViT);
