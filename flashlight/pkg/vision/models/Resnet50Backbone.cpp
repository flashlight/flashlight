/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/models/Resnet50Backbone.h"

namespace fl::pkg::vision {

Resnet50Backbone::Resnet50Backbone() {
  Sequential backbone;
  backbone.add(ConvFrozenBatchNormActivation(3, 64, 7, 7, 2, 2));
  // maxpool -> 112x122x64 -> 56x56x64
  backbone.add(Pool2D(3, 3, 2, 2, -1, -1, PoolingMode::MAX));
  // conv2_x -> 56x56x64 -> 56x56x64
  backbone.add(ResNetBottleneckStageFrozenBatchNorm(64, 64, 3, 1));
  // conv3_x -> 56x56x64 -> 28x28x128
  backbone.add(ResNetBottleneckStageFrozenBatchNorm(64 * 4, 128, 4, 2));
  // conv4_x -> 28x28x128 -> 14x14x256
  backbone.add(ResNetBottleneckStageFrozenBatchNorm(128 * 4, 256, 6, 2));
  // conv5_x -> 14x14x256 -> 7x7x256
  backbone.add(ResNetBottleneckStageFrozenBatchNorm(256 * 4, 512, 3, 2));
  add(std::move(backbone));

  Sequential tail;
  tail.add(Pool2D(7, 7, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));
  tail.add(
      ConvFrozenBatchNormActivation(512 * 4, 1000, 1, 1, 1, 1, false, false));
  tail.add(View({1000, -1}));
  tail.add(LogSoftmax());
  add(std::move(tail));
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

} // namespace fl
