/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/nn/nn.h"

namespace fl {
namespace ext {
namespace image {

class ConvBnAct : public fl::Sequential {
 public:
  ConvBnAct();
  explicit ConvBnAct(
      const int in_channels,
      const int out_channels,
      const int kw,
      const int kh,
      const int sx = 1,
      const int sy = 1,
      bool bn = true,
      bool act = true);

 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};

class ResNetBlock : public fl::Container {
 private:
   FL_SAVE_LOAD_WITH_BASE(fl::Container)
 public:
  ResNetBlock();
  explicit ResNetBlock(
      const int in_channels,
      const int out_channels,
      const int stride=1);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  std::string prettyString() const override;

};

class ResNetStage : public fl::Sequential {
 public:
  ResNetStage();
  explicit ResNetStage(
      const int in_channels,
      const int out_channels,
      const int num_blocks,
      const int stride);
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};


Sequential resnet34();


} // namespace image
} // namespace ext
} // namespace fl
CEREAL_REGISTER_TYPE(fl::ext::image::ConvBnAct)
CEREAL_REGISTER_TYPE(fl::ext::image::ResNetBlock)
CEREAL_REGISTER_TYPE(fl::ext::image::ResNetStage)
