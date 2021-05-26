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
namespace pkg {
namespace vision {

// Note these are identical to those in Resnet.h. There are a number of ways to
// refactor and consolidate including passing norm factory functions to the
// constructor or templating the class. However, for the sake of keeping
// the default Resnet implementation dead simple, we are recreating a lot
// of functionality here.

class ConvFrozenBatchNormActivation : public fl::Sequential {
 public:
  ConvFrozenBatchNormActivation(
      const int inChannels,
      const int outChannels,
      const int kw,
      const int kh,
      const int sx = 1,
      const int sy = 1,
      bool bn = true,
      bool act = true);

 private:
  ConvFrozenBatchNormActivation();
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};

class ResNetBlockFrozenBatchNorm : public fl::Container {
 private:
  ResNetBlockFrozenBatchNorm();
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
 public:
  ResNetBlockFrozenBatchNorm(
      const int inChannels,
      const int outChannels,
      const int stride = 1);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  std::string prettyString() const override;
};

class ResNetBottleneckBlockFrozenBatchNorm : public fl::Container {
 private:
  ResNetBottleneckBlockFrozenBatchNorm();
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
 public:
  ResNetBottleneckBlockFrozenBatchNorm(
      const int inChannels,
      const int outChannels,
      const int stride = 1);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  std::string prettyString() const override;
};

class ResNetBottleneckStageFrozenBatchNorm : public fl::Sequential {
 public:
  ResNetBottleneckStageFrozenBatchNorm(
      const int inChannels,
      const int outChannels,
      const int numBlocks,
      const int stride);

 private:
  ResNetBottleneckStageFrozenBatchNorm();
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};

class ResNetStageFrozenBatchNorm : public fl::Sequential {
 public:
  ResNetStageFrozenBatchNorm(
      const int inChannels,
      const int outChannels,
      const int numBlocks,
      const int stride);

 private:
  ResNetStageFrozenBatchNorm();
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};

} // namespace vision
} // namespace pkg
} // namespace fl
CEREAL_REGISTER_TYPE(fl::pkg::vision::ConvFrozenBatchNormActivation)
CEREAL_REGISTER_TYPE(fl::pkg::vision::ResNetBlockFrozenBatchNorm)
CEREAL_REGISTER_TYPE(fl::pkg::vision::ResNetStageFrozenBatchNorm)
CEREAL_REGISTER_TYPE(fl::pkg::vision::ResNetBottleneckBlockFrozenBatchNorm)
CEREAL_REGISTER_TYPE(fl::pkg::vision::ResNetBottleneckStageFrozenBatchNorm)
