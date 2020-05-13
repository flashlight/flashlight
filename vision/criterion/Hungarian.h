#pragma once

#include "flashlight/nn/modules/Container.h"

namespace fl {
namespace cv {

class HungarianMatcher {

public:
  HungarianMatcher() = default;

  HungarianMatcher(
      const float cost_class,
      const float cost_bbox,
      const float cost_giou);

  std::vector<std::pair<af::array, af::array>> forward(
      const af::array& predBoxes,
      const af::array& predLogits,
      const af::array& targetBoxes,
      const af::array& targetClasses) const;


private:
  float cost_class_;
  float cost_bbox_;
  float cost_giou_;

   std::pair<af::array, af::array> matchBatch(
       const af::array& predBoxes, 
       const af::array& predLogits,
       const af::array& targetBoxes, 
       const af::array& targetClasses) const;

  af::array getCostMatrix(
      const af::array& input, 
      const af::array& target);

};

} // cv
} // flashlight
