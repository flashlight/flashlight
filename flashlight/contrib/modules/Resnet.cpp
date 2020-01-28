#include "flashlight/contrib/modules/Resnet.h"

namespace fl {

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

ResNetBlock::ResNetBlock(const int in_c, const int out_c, bool downsample)
    : downsample_(downsample) {
  const int stride = (downsample_) ? 2 : 1;
  add(std::make_shared<ConvBnAct>(in_c, out_c, 3, 3, stride, stride));
  add(std::make_shared<ConvBnAct>(out_c, out_c, 3, 3, 1, 1, true, false));
  add(std::make_shared<ReLU>());
}

std::vector<fl::Variable> ResNetBlock::forward(
    const std::vector<fl::Variable>& inputs) {
  auto c1 = module(0);
  auto c2 = module(1);
  auto relu = module(2);
  auto out = c1->forward(inputs);
  out = c2->forward(out);
  if (downsample_) {
    return relu->forward(out);
  } else {
    return {relu->forward({out[0] + inputs[0]})};
  }
}

std::string ResNetBlock::prettyString() const {
  return "2-Layer ResNetBlock Conv3x3";
}

template <class Archive>
void ResNetBlock::serialize(Archive& ar) {
  ar(cereal::base_class<Container>(this));
}

ResNetStage::ResNetStage(
    const int in_c,
    const int out_c,
    const int num_blocks,
    bool downsample) {
  add(ResNetBlock(in_c, out_c, downsample));
  for (int i = 1; i < num_blocks; i++) {
    add(ResNetBlock(out_c, out_c, false));
  }
}

Sequential resnet34() {
  Sequential model;
  // conv1 -> 244x244x3 -> 112x112x64
  model.add(ConvBnAct(3, 64, 7, 7, 2, 2));
  // maxpool -> 112x122x64 -> 56x56x64
  model.add(Pool2D(3, 3, 2, 2, -1, -1));
  // conv2_x -> 56x56x64 -> 56x56x64
  model.add(ResNetStage(64, 64, 3, false));
  // conv3_x -> 56x56x64 -> 28x28x128
  model.add(ResNetStage(64, 128, 4, true));
  // conv4_x -> 28x28x128 -> 14x14x256
  model.add(ResNetStage(128, 256, 6, true));
  // conv5_x -> 14x14x256 -> 7x7x256
  model.add(ResNetStage(256, 512, 3, true));
  // pool 7x7x64 ->
  model.add(Pool2D(7, 7, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));
  model.add(ConvBnAct(512, 1000, 1, 1, 1, 1, false, false));
  model.add(View({1000, -1}));
  model.add(LogSoftmax());
  return model;
};

}; // namespace fl
