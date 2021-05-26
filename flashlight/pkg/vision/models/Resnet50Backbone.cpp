/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/models/Resnet50Backbone.h"

namespace fl {
namespace pkg {
namespace vision {

Resnet50Backbone::Resnet50Backbone()
    : backbone_(std::make_shared<Sequential>()),
      tail_(std::make_shared<Sequential>()) {
  backbone_->add(ConvFrozenBatchNormActivation(3, 64, 7, 7, 2, 2));
  // maxpool -> 112x122x64 -> 56x56x64
  backbone_->add(Pool2D(3, 3, 2, 2, -1, -1, PoolingMode::MAX));
  // conv2_x -> 56x56x64 -> 56x56x64
  backbone_->add(ResNetBottleneckStageFrozenBatchNorm(64, 64, 3, 1));
  // conv3_x -> 56x56x64 -> 28x28x128
  backbone_->add(ResNetBottleneckStageFrozenBatchNorm(64 * 4, 128, 4, 2));
  // conv4_x -> 28x28x128 -> 14x14x256
  backbone_->add(ResNetBottleneckStageFrozenBatchNorm(128 * 4, 256, 6, 2));
  // conv5_x -> 14x14x256 -> 7x7x256
  backbone_->add(ResNetBottleneckStageFrozenBatchNorm(256 * 4, 512, 3, 2));

  tail_->add(Pool2D(7, 7, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));
  tail_->add(
      ConvFrozenBatchNormActivation(512 * 4, 1000, 1, 1, 1, 1, false, false));
  tail_->add(View({1000, -1}));
  tail_->add(LogSoftmax());
  add(backbone_);
  add(tail_);
}

std::vector<Variable> Resnet50Backbone::forward(
    const std::vector<Variable>& input) {
  const auto& features = module(0)->forward(input);
  const auto& output = module(1)->forward(features);
  return {output[0], features[0]};
}

std::string Resnet50Backbone::prettyString() const {
  return "Resnet50Backbone";
}

} // namespace vision
} // namespace pkg
} // namespace fl
