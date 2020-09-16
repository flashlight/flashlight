#pragma once

#include <iostream>

#include "models/Resnet.h"

namespace fl {
namespace cv {

// TODO HACK!!!!!!
void freezeBatchNorm(std::shared_ptr<fl::Module> ptr) {
  //std::cout << "Freezing " << ptr->prettyString() << std::endl;
  if(dynamic_cast<fl::BatchNorm*>(ptr.get())) {
    ptr->eval();
  } else if(dynamic_cast<fl::Container*>(ptr.get())) {
      for(auto mod : dynamic_cast<fl::Container*>(ptr.get())->modules()) {
        freezeBatchNorm(mod);
      }
  }
}

//class Joiner: public Container {

  //Joiner(std::shared_ptr<Module> base) :
    //base_(base) {
      //add(base_);
    //}

  //std::vector<Variable> forward(const std::vector<Variable&> inputs) {
    //auto images = inputs[0];
    //auto masks = inputs[1];
    //auto outputs = base_->forward(images);
    //std::vector<Variable> results;
    //for(auto output : outputs) {
      //results.push_back(output);
      //auto resized_mask = af::resize(masks, output.dim(0), output.dim(1));
      //results.push_back(fl::Variable(resized_mask, false));
    //}
  //}

  //private:
    //std::shared_ptr<Module> base_;
//};

// TODO Can probably generalize this
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
