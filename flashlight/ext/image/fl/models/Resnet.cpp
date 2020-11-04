/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/image/fl/models/Resnet.h"

namespace fl {
namespace ext {
namespace image {

namespace {

Conv2D conv3x3(int inC, int outC, int stride, int groups) {
  const auto pad = PaddingMode::SAME;
  return Conv2D(
      inC, outC, 3, 3, stride, stride, pad, pad, 1, 1, false, groups);
}

Conv2D conv1x1(int inC, int outC, int stride, int groups) {
  const auto pad = PaddingMode::SAME;
  return Conv2D(
      inC, outC, 1, 1, stride, stride, pad, pad, 1, 1, false, groups);
}

} // namespace

ConvBnAct::ConvBnAct() = default;

ConvBnAct::ConvBnAct(
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
  add(std::make_shared<fl::Conv2D>(
      inC, outC, kw, kh, sx, sy, pad, pad, 1, 1, bias));
  if (bn) {
    add(std::make_shared<fl::BatchNorm>(2, outC));
  }
  if (act) {
    add(std::make_shared<fl::ReLU>());
  }
}

ResNetBlock::ResNetBlock() = default;

ResNetBlock::ResNetBlock(const int inC, const int outC, const int stride) {
  add(std::make_shared<Conv2D>(conv3x3(inC, outC, stride, 1)));
  add(std::make_shared<BatchNorm>(BatchNorm(2, outC)));
  add(std::make_shared<ReLU>());
  add(std::make_shared<Conv2D>(conv3x3(outC, outC, 1, 1)));
  add(std::make_shared<BatchNorm>(BatchNorm(2, outC)));
  add(std::make_shared<ReLU>());
  if (inC != outC || stride > 1) {
    Sequential downsample;
    downsample.add(conv1x1(inC, outC, stride, 1));
    downsample.add(BatchNorm(2, outC));
    add(downsample);
  }
}

std::vector<fl::Variable> ResNetBlock::forward(
    const std::vector<fl::Variable>& inputs) {
  auto c1 = module(0);
  auto bn1 = module(1);
  auto relu1 = module(2);
  auto c2 = module(3);
  auto bn2 = module(4);
  auto relu2 = module(5);
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

std::string ResNetBlock::prettyString() const {
  return "2-Layer ResNetBlock Conv3x3";
}

ResNetStage::ResNetStage() = default;

ResNetStage::ResNetStage(
    const int inC,
    const int outC,
    const int numBlocks,
    const int stride) {
  add(ResNetBlock(inC, outC, stride));
  for (int i = 1; i < numBlocks; i++) {
    add(ResNetBlock(outC, outC));
  }
}

std::shared_ptr<Sequential> resnet34() {
  auto model = std::make_shared<Sequential>();
  // conv1 -> 244x244x3 -> 112x112x64
  model->add(ConvBnAct(3, 64, 7, 7, 2, 2));
  // maxpool -> 112x122x64 -> 56x56x64
  model->add(Pool2D(3, 3, 2, 2, -1, -1, PoolingMode::MAX));
  // conv2_x -> 56x56x64 -> 56x56x64
  model->add(ResNetStage(64, 64, 3, 1));
  // conv3_x -> 56x56x64 -> 28x28x128
  model->add(ResNetStage(64, 128, 4, 2));
  // conv4_x -> 28x28x128 -> 14x14x256
  model->add(ResNetStage(128, 256, 6, 2));
  // conv5_x -> 14x14x256 -> 7x7x256
  model->add(ResNetStage(256, 512, 3, 2));
  // pool 7x7x512 -> 1x1x512
  model->add(Pool2D(7, 7, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));
  model->add(ConvBnAct(512, 1000, 1, 1, 1, 1, false, false));
  model->add(View({1000, -1}));
  model->add(LogSoftmax());
  return model;
};

} // namespace image
} // namespace ext
} // namespace fl
