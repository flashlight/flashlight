#include "flashlight/contrib/modules/Resnet.h"


namespace fl {

namespace {

Conv2D conv3x3(const int in_c, const int out_c, const int stride,
    const int groups) {
  const auto pad = PaddingMode::SAME;
  return Conv2D(in_c, out_c, 3, 3, stride, stride, pad, pad, 1, 1, false);
}

Conv2D conv1x1(const int in_c, const int out_c, const int stride,
    const int groups) {
  const auto pad = PaddingMode::SAME;
  return Conv2D(in_c, out_c, 1, 1, stride, stride, pad, pad, 1, 1, false);
}

}

ConvBnAct::ConvBnAct() = default;

ConvBnAct::ConvBnAct(
    const int in_c,
    const int out_c,
    const int kw,
    const int kh,
    const int sx,
    const int sy,
    bool bn,
    bool act) {
  const auto pad = PaddingMode::SAME;
  const bool bias = !bn;
  add(std::make_shared<fl::Conv2D>(
      in_c, out_c, kw, kh, sx, sy, pad, pad, 1, 1, bias));
  if (bn) {
    add(std::make_shared<fl::BatchNorm>(2, out_c));
  }
  if (act) {
    add(std::make_shared<fl::ReLU>());
  }
}

ResNetBlock::ResNetBlock() = default;

ResNetBlock::ResNetBlock(const int in_c, const int out_c, const int stride) {
  add(std::make_shared<Conv2D>(conv3x3(in_c, out_c, stride, 1)));
  add(std::make_shared<BatchNorm>(BatchNorm(2, out_c)));
  add(std::make_shared<ReLU>());
  add(std::make_shared<Conv2D>(conv3x3(out_c, out_c, 1, 1)));
  add(std::make_shared<BatchNorm>(BatchNorm(2, out_c)));
  add(std::make_shared<ReLU>());
  if (in_c != out_c || stride == 2) {
    Sequential downsample;
    downsample.add(conv1x1(in_c, out_c, stride, 1));
    downsample.add(BatchNorm(2, out_c));
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
    const int in_c,
    const int out_c,
    const int num_blocks,
    const int stride) {
  add(ResNetBlock(in_c, out_c, stride));
  for (int i = 1; i < num_blocks; i++) {
    add(ResNetBlock(out_c, out_c));
  }
}

Sequential resnet34() {
  Sequential model;
  // conv1 -> 244x244x3 -> 112x112x64
  model.add(ConvBnAct(3, 64, 7, 7, 2, 2));
  // maxpool -> 112x122x64 -> 56x56x64
  model.add(Pool2D(3, 3, 2, 2, -1, -1, PoolingMode::MAX));
  // conv2_x -> 56x56x64 -> 56x56x64
  model.add(ResNetStage(64, 64, 3, 1));
  // conv3_x -> 56x56x64 -> 28x28x128
  model.add(ResNetStage(64, 128, 4, 2));
  // conv4_x -> 28x28x128 -> 14x14x256
  model.add(ResNetStage(128, 256, 6, 2));
  // conv5_x -> 14x14x256 -> 7x7x256
  model.add(ResNetStage(256, 512, 3, 2));
  // pool 7x7x64 ->
  model.add(Pool2D(7, 7, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));
  model.add(ConvBnAct(512, 1000, 1, 1, 1, 1, false, false));
  model.add(View({1000, -1}));
  model.add(LogSoftmax());
  return model;
};

}; // namespace fl
