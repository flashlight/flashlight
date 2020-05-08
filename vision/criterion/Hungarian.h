#pragma once

#include "flashlight/nn/modules/Container.h"

namespace fl {
namespace cv {

class HungarianMatcher {

public:
  HungarianMatcher() = default;
  HungarianMatcher(const float cost_class, const float cost_bbox, const float cost_giou);

  std::vector<std::pair<af::array, af::array>> forward(const af::array& predBoxes, const af::array& predLogits,
      const af::array& targetBoxes, const af::array& targetClasses) const;


private:

  af::array getCostMatrix(const af::array& input, const af::array& target);
  float cost_class_;
  float cost_bbox_;
  float cost_giou;
};

//class HungarianLoss : public BinaryModule {

//public:
  //HungarianLoss(const float cost_class, const float cost_bbox, const float cost_giou);

  //Variable forward(const Variable& input, const Variable& target) override;

  //virtual std::string prettyString() const override;

//private:
  //HungarianMatcher matcher;
//};

} // cv
} // flashlight
