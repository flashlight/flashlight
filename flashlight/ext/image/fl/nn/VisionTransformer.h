/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/nn.h"

namespace fl {
namespace ext {
namespace image {

/*
 * Implementation of the transformer blocks of Vision Transformer (ViT) models
 * following [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION
 * AT SCALE](https://arxiv.org/pdf/2010.11929.pdf)
 *
 * This implementation is highly inspired by [timm](https://git.io/JYOql).
 */
class VisionTransformer : public Container {
 public:
  VisionTransformer(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      float pDropout,
      float pLayerdrop);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;
  std::string prettyString() const override;

 private:
  int32_t modelDim_;
  int32_t headDim_;
  int32_t mlpDim_;
  int32_t nHeads_;
  double pDropout_;
  double pLayerdrop_;
  std::shared_ptr<Linear> w1_, w2_;
  std::shared_ptr<Linear> wq_, wk_, wv_;
  std::shared_ptr<Linear> wf_;
  std::shared_ptr<LayerNorm> norm1_, norm2_;

  Variable gelu(const Variable& input);
  Variable mlp(const Variable& input);
  Variable selfAttention(const Variable& input);
  Variable dropPath(const Variable& input);

  FL_SAVE_LOAD_WITH_BASE(
      Container,
      w1_,
      w2_,
      wq_,
      wk_,
      wv_,
      wf_,
      norm1_,
      norm2_,
      modelDim_,
      headDim_,
      mlpDim_,
      nHeads_,
      pDropout_,
      pLayerdrop_)

  VisionTransformer() = default;

  std::shared_ptr<fl::Linear> initLinear(int inDim, int outDim);
};

} // namespace image
} // namespace ext
} // namespace fl

CEREAL_REGISTER_TYPE(fl::ext::image::VisionTransformer);
