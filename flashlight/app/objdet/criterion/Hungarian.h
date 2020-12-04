#pragma once

#include "flashlight/fl/nn/modules/Container.h"

namespace fl {
namespace app {
namespace objdet {

class HungarianMatcher {

public:
  HungarianMatcher() = default;

  HungarianMatcher(
      const float cost_class,
      const float cost_bbox,
      const float cost_giou);

  std::vector<std::pair<af::array, af::array>> forward(
      const Variable& predBoxes,
      const Variable& predLogits,
      const std::vector<Variable>& targetBoxes,
      const std::vector<Variable>& targetClasses) const;


private:
  float cost_class_;
  float cost_bbox_;
  float cost_giou_;

  // First is SrcIdx, second is ColIdx
   std::pair<af::array, af::array> matchBatch(
       const Variable& predBoxes,
       const Variable& predLogits,
       const Variable& targetBoxes,
       const Variable& targetClasses) const;

  af::array getCostMatrix(
      const Variable& input,
      const Variable& target);

};

} // objdet
} // app
} // fl
