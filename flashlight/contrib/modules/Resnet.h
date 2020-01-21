#pragma once

#include "flashlight/nn/nn.h"

namespace fl {

class ConvBnAct : public Sequential {
 public:
  explicit ConvBnAct(
      const int in_channels,
      const int out_channels,
      const int kw,
      const int kh,
      const int sx = 1,
      const int sy = 1,
      bool bn = true,
      bool act = true);
};

class ResNetBlock : public Container {
 public:
  explicit ResNetBlock(
      const int in_channels,
      const int out_channels,
      bool downsample = false);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  std::string prettyString() const override;

  template <class Archive>
  void serialize(Archive& ar);

 private:
  bool downsample_;
};

class ResNetStage : public Sequential {
 public:
  explicit ResNetStage(
      const int in_channels,
      const int out_channels,
      const int num_blocks,
      bool downsample);
};

Sequential resnet34();

} // namespace fl
