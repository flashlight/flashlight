/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/vision/nn/FrozenBatchNorm.h"
#include "flashlight/fl/nn/nn.h"

namespace fl {
namespace ext {
namespace image {

class ConvBnAct : public fl::Sequential {
 public:
  ConvBnAct(
      const int inChannels,
      const int outChannels,
      const int kw,
      const int kh,
      const int sx = 1,
      const int sy = 1,
      bool bn = true,
      bool act = true);

 private:
  ConvBnAct();
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};

class ResNetBlock : public fl::Container {
 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
  ResNetBlock();

 public:
  ResNetBlock(
      const int inChannels,
      const int outChannels,
      const int stride = 1);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  std::string prettyString() const override;
};

class ResNetBottleneckBlock : public fl::Container {
 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
  ResNetBottleneckBlock();

 public:
  ResNetBottleneckBlock(
      const int inChannels,
      const int outChannels,
      const int stride = 1);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  std::string prettyString() const override;
};

class ResNetBottleneckStage : public fl::Sequential {
 public:
  ResNetBottleneckStage(
      const int inChannels,
      const int outChannels,
      const int numBlocks,
      const int stride);

 private:
  ResNetBottleneckStage();
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};

class ResNetStage : public fl::Sequential {
 public:
  ResNetStage(
      const int inChannels,
      const int outChannels,
      const int numBlocks,
      const int stride);

 private:
  ResNetStage();
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};

std::shared_ptr<Sequential> resnet34();
std::shared_ptr<Sequential> resnet50();

} // namespace image
} // namespace ext
} // namespace fl
CEREAL_REGISTER_TYPE(fl::ext::image::ConvBnAct)
CEREAL_REGISTER_TYPE(fl::ext::image::ResNetBlock)
CEREAL_REGISTER_TYPE(fl::ext::image::ResNetStage)
CEREAL_REGISTER_TYPE(fl::ext::image::ResNetBottleneckBlock)
CEREAL_REGISTER_TYPE(fl::ext::image::ResNetBottleneckStage)
