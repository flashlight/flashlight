/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/models/ResnetFrozenBatchNorm.h"

namespace fl {
namespace pkg {
namespace vision {

namespace {

Conv2D conv3x3(int inC, int outC, int stride, int groups) {
  const auto pad = PaddingMode::SAME;
  return Conv2D(inC, outC, 3, 3, stride, stride, pad, pad, 1, 1, false, groups);
}

Conv2D conv1x1(int inC, int outC, int stride, int groups) {
  const auto pad = PaddingMode::SAME;
  return Conv2D(inC, outC, 1, 1, stride, stride, pad, pad, 1, 1, false, groups);
}

} // namespace

ConvFrozenBatchNormActivation::ConvFrozenBatchNormActivation() = default;

ConvFrozenBatchNormActivation::ConvFrozenBatchNormActivation(
    const int inC,
    const int outC,
    const int kw,
    const int kh,
    const int sx,
    const int sy,
    bool bn,
    bool act) {
  const auto pad = PaddingMode::SAME;
  const bool bias = !bn;
  add(fl::Conv2D(inC, outC, kw, kh, sx, sy, pad, pad, 1, 1, bias));
  if (bn) {
    add(fl::FrozenBatchNorm(2, outC));
  }
  if (act) {
    add(fl::ReLU());
  }
}

ResNetBlockFrozenBatchNorm::ResNetBlockFrozenBatchNorm() = default;

ResNetBlockFrozenBatchNorm::ResNetBlockFrozenBatchNorm(
    const int inC,
    const int outC,
    const int stride) {
  add(Conv2D(conv3x3(inC, outC, stride, 1)));
  add(FrozenBatchNorm(FrozenBatchNorm(2, outC)));
  add(ReLU());
  add(conv3x3(outC, outC, 1, 1));
  add(FrozenBatchNorm(2, outC));
  add(ReLU());
  if (inC != outC || stride > 1) {
    Sequential downsample;
    downsample.add(conv1x1(inC, outC, stride, 1));
    downsample.add(FrozenBatchNorm(2, outC));
    add(downsample);
  }
}

ResNetBottleneckBlockFrozenBatchNorm::ResNetBottleneckBlockFrozenBatchNorm() =
    default;

ResNetBottleneckBlockFrozenBatchNorm::ResNetBottleneckBlockFrozenBatchNorm(
    const int inC,
    const int planes,
    const int stride) {
  const int expansionFactor = 4;
  add(conv1x1(inC, planes, 1, 1));
  add(FrozenBatchNorm(FrozenBatchNorm(2, planes)));
  add(ReLU());
  add(Conv2D(conv3x3(planes, planes, stride, 1)));
  add(FrozenBatchNorm(FrozenBatchNorm(2, planes)));
  add(ReLU());
  add(conv1x1(planes, planes * expansionFactor, 1, 1));
  add(FrozenBatchNorm(2, planes * expansionFactor));
  add(ReLU());
  if (inC != planes * expansionFactor || stride > 1) {
    Sequential downsample;
    downsample.add(conv1x1(inC, planes * expansionFactor, stride, 1));
    downsample.add(FrozenBatchNorm(2, planes * expansionFactor));
    add(std::move(downsample));
  }
}

std::vector<fl::Variable> ResNetBottleneckBlockFrozenBatchNorm::forward(
    const std::vector<fl::Variable>& inputs) {
  const auto& c1 = module(0);
  const auto& bn1 = module(1);
  const auto& relu1 = module(2);
  const auto& c2 = module(3);
  const auto& bn2 = module(4);
  const auto& relu2 = module(5);
  const auto& c3 = module(6);
  const auto& bn3 = module(7);
  const auto& relu3 = module(8);

  std::vector<fl::Variable> out;
  out = c1->forward(inputs);
  out = bn1->forward(out);

  out = relu1->forward(out);

  out = c2->forward(out);
  out = bn2->forward(out);
  out = relu2->forward(out);

  out = c3->forward(out);
  out = bn3->forward(out);

  std::vector<fl::Variable> shortcut;
  if (modules().size() > 9) {
    shortcut = module(9)->forward(inputs);
  } else {
    shortcut = inputs;
  }
  return relu3->forward({out[0] + shortcut[0]});
}

std::string ResNetBottleneckBlockFrozenBatchNorm::prettyString() const {
  std::ostringstream ss;
  ss << "ResnetBottleneckBlockFrozenBn";
  ss << Container::prettyString();
  return ss.str();
}

std::vector<fl::Variable> ResNetBlockFrozenBatchNorm::forward(
    const std::vector<fl::Variable>& inputs) {
  const auto& c1 = module(0);
  const auto& bn1 = module(1);
  const auto& relu1 = module(2);
  const auto& c2 = module(3);
  const auto& bn2 = module(4);
  const auto& relu2 = module(5);
  std::vector<fl::Variable> out;
  out = c1->forward(inputs);
  out = bn1->forward(out);
  out = relu1->forward(out);
  out = c2->forward(out);
  out = bn2->forward(out);

  std::vector<fl::Variable> shortcut;
  if (modules().size() > 6) {
    shortcut = module(6)->forward(inputs);
  } else {
    shortcut = inputs;
  }
  return relu2->forward({out[0] + shortcut[0]});
}

std::string ResNetBlockFrozenBatchNorm::prettyString() const {
  std::ostringstream ss;
  ss << "ResnetBlockFrozenBn";
  ss << Container::prettyString();
  return ss.str();
}

ResNetBottleneckStageFrozenBatchNorm::ResNetBottleneckStageFrozenBatchNorm(
    const int inC,
    const int outC,
    const int numBlocks,
    const int stride) {
  add(ResNetBottleneckBlockFrozenBatchNorm(inC, outC, stride));
  const int expansionFactor = 4;
  const int inPlanes = outC * expansionFactor;
  for (int i = 1; i < numBlocks; i++) {
    add(ResNetBottleneckBlockFrozenBatchNorm(inPlanes, outC));
  }
};

ResNetBottleneckStageFrozenBatchNorm::ResNetBottleneckStageFrozenBatchNorm() =
    default;

ResNetStageFrozenBatchNorm::ResNetStageFrozenBatchNorm() = default;

ResNetStageFrozenBatchNorm::ResNetStageFrozenBatchNorm(
    const int inC,
    const int outC,
    const int numBlocks,
    const int stride) {
  add(ResNetBlockFrozenBatchNorm(inC, outC, stride));
  for (int i = 1; i < numBlocks; i++) {
    add(ResNetBlockFrozenBatchNorm(outC, outC));
  }
}

} // namespace vision
} // namespace pkg
} // namespace fl
