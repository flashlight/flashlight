#pragma once

#include "flashlight/nn/nn.h"

namespace fl {

class ConvBnAct : public Sequential {
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
  FL_SAVE_LOAD_WITH_BASE(Sequential)
};


class ResNetBlock : public Container {
 private:
  bool downsample_;
  FL_SAVE_LOAD_WITH_BASE(Container)
 public:
  ResNetBlock();
  explicit ResNetBlock(
      const int in_channels,
      const int out_channels,
      bool downsample = false);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  std::string prettyString() const override;

};

//CEREAL_REGISTER_TYPE(ResNetBlock)

class ResNetStage : public Sequential {
 public:
  ResNetStage();
  explicit ResNetStage(
      const int in_channels,
      const int out_channels,
      const int num_blocks,
      bool downsample);
  FL_SAVE_LOAD_WITH_BASE(Sequential)
};


Sequential resnet34();

} // namespace fl
CEREAL_REGISTER_TYPE(fl::ConvBnAct)
CEREAL_REGISTER_TYPE(fl::ResNetBlock)
CEREAL_REGISTER_TYPE(fl::ResNetStage)
