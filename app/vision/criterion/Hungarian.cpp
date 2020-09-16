#include "Hungarian.h"
#include "vision/dataset/BoxUtils.h"
#include "vision/criterion/HungarianLib.h"

#include "flashlight/autograd/Functions.h"

namespace {

std::pair<af::array, af::array> hungarian(af::array& cost) {
    cost = cost.T();
    //af_print(cost);
    const int M = cost.dims(0);
    const int N = cost.dims(1);
    std::vector<float> costHost(cost.elements());
    std::vector<int> rowIdxs(M);
    std::vector<int> colIdxs(M);
    cost.host(costHost.data());
    fl::cv::hungarian(costHost.data(), rowIdxs.data(), colIdxs.data(), M, N);
    auto rowIdxsArray = af::array(M, rowIdxs.data());
    auto colIdxsArray = af::array(M, colIdxs.data());
    return { rowIdxsArray, colIdxsArray };
  }
}

namespace fl {
namespace cv {

HungarianMatcher::HungarianMatcher(
    const float cost_class,
    const float cost_bbox,
    const float cost_giou) :
  cost_class_(cost_class), cost_bbox_(cost_bbox), cost_giou_(cost_giou) {
};

std::pair<af::array, af::array> HungarianMatcher::matchBatch(
    const Variable& predBoxes, 
    const Variable& predLogits,
    const Variable& targetBoxes, 
    const Variable& targetClasses) const {

  // TODO Kind of a hack...
  if(targetClasses.isempty()) {
    return { af::array(0, 1), af::array(0, 1) };
  }


  // Create an M X N cost matrix where M is the number of targets and N is the number of preds
  
  // Class cost
  auto outProbs = softmax(predLogits, 0);
  auto cost_class = transpose((1 - outProbs(targetClasses.array(), af::span)));
  //auto cost_class = (1 - outProbs(targetClasses.array(), af::span));
  //


  // Generalized IOU loss
  auto cost_giou =  0 - dataset::generalized_box_iou(
      dataset::cxcywh_to_xyxy(predBoxes), 
      dataset::cxcywh_to_xyxy(targetBoxes)
  );

  // Bbox Cost
  Variable cost_bbox = dataset::cartesian(predBoxes, targetBoxes,
      [](const Variable& x, const Variable& y) {
        return sum(abs(x - y), {0});
    });
  cost_bbox = dataset::flatten(cost_bbox, 0, 1);

  auto cost = cost_bbox_ * cost_bbox + cost_class_ * cost_class + cost_giou_ * cost_giou;
  return ::hungarian(cost.array());


}

std::vector<std::pair<af::array, af::array>> HungarianMatcher::forward(
    const Variable& predBoxes,
    const Variable& predLogits,
    const std::vector<Variable>& targetBoxes,
    const std::vector<Variable>& targetClasses) const {
  std::vector<std::pair<af::array, af::array>> results;
  for(int b = 0; b < predBoxes.dims(2); b++) {
    auto result = matchBatch(
      predBoxes(af::span, af::span, b),
      predLogits(af::span, af::span, b),
      targetBoxes[b],
      targetClasses[b]);
    results.emplace_back(result);
  }
  return results;
};

} // end namespace cv
} // end namespace flashlight


