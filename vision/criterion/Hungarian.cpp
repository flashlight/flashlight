#include "Hungarian.h"



namespace fl {
namespace cv {

HungarianMatcher::HungarianMatcher(
    const float cost_class, 
    const float cost_bbox, 
    const float cost_giou) :
  cost_class_(cost_class), cost_bbox_(cost_bbox), cost_giou(cost_giou) {
};

Variable HungarianMatcher::forward(const Variable& input, const Variable& target) {
  return input;
};

std::string HungarianMatcher::prettyString() const {
  return "HungarianMatcher";
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


