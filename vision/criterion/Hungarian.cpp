#include "Hungarian.h"


float bbox_loss(const af::array& input, const af::array& target) {
  return (target - input).sum();
}

namespace fl {
namespace cv {

HungarianMatcher::HungarianMatcher(
    const float cost_class, 
    const float cost_bbox, 
    const float cost_giou) :
  cost_class_(cost_class), cost_bbox_(cost_bbox), cost_giou(cost_giou) {
};

Variable HungarianMatcher::forward(const Variable& input, const Variable& target) {
  costs = getCostMatrix(input.array(), target.array());

  return input;
};

std::string HungarianMatcher::prettyString() const {
  return "HungarianMatcher";
}

af::array getCostMatrix(const af::array& input, const af::array& target) {
  // input [5, N]
  // target [5, N]
  cost = af::constant(0, input.dims(1), target.dims(1));
  for(int i = 0; i < input.dims(1); i++) {
    af::array avec = input(af::span, i);
    for(int j = 0; j < target.dims(1); j++) {
      array bvec = b(span, j);
      cost(i, j) = bbox_loss(avec, bvec);
    }
  }
}
  
}



//HungarianLoss::HungarianLoss(
    //const float cost_class, 
    //const float cost_bbox, 
    //const float cost_giou) :
  //matcher_(cost_class, cost_bbox, cost_giou) {
//};

//Variable forward(const Variable& input, const Variable& target) {
  //return input;
//};

} // end namespace cv
} // end namespace flashlight


