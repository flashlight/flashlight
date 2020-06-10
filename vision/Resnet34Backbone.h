#pragma once

#include "models/Resnet.h"

namespace fl {
namespace cv {

class Resnet34Backbone : public Container {

public:
  Resnet34Backbone() : 
    backbone_(std::make_shared<Sequential>()),
    tail_(std::make_shared<Sequential>())
  {

    backbone_->add(ConvBnAct(3, 64, 7, 7, 2, 2));
    // maxpool -> 112x122x64 -> 56x56x64
    backbone_->add(Pool2D(3, 3, 2, 2, -1, -1, PoolingMode::MAX));
    // conv2_x -> 56x56x64 -> 56x56x64
    backbone_->add(ResNetStage(64, 64, 3, 1));
    // conv3_x -> 56x56x64 -> 28x28x128
    backbone_->add(ResNetStage(64, 128, 4, 2));
    // conv4_x -> 28x28x128 -> 14x14x256
    backbone_->add(ResNetStage(128, 256, 6, 2));
    // conv5_x -> 14x14x256 -> 7x7x256
    backbone_->add(ResNetStage(256, 512, 3, 2));

    tail_->add(Pool2D(7, 7, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));
    tail_->add(ConvBnAct(512, 1000, 1, 1, 1, 1, false, false));
    tail_->add(View({1000, -1}));
    tail_->add(LogSoftmax());
    add(backbone_);
    add(tail_);
  }

  std::vector<Variable> forward(const std::vector<Variable>& input) {
    auto features = backbone_->forward(input);
    auto output = tail_->forward(features);
    return { output[0], features[0] };
  }

  std::string prettyString() const {
    return "Resnet34 Backbone";
  }

 private:
   std::shared_ptr<Sequential> backbone_;
   std::shared_ptr<Sequential> tail_;
    FL_SAVE_LOAD_WITH_BASE(fl::Container)
};

} // end namespace cv
} // end namespace fl

CEREAL_REGISTER_TYPE(fl::cv::Resnet34Backbone)
