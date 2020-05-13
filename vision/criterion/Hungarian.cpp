#include "Hungarian.h"
#include "vision/dataset/BoxUtils.h"
#include "vision/criterion/HungarianLib.h"

namespace {

std::pair<af::array, af::array> hungarian(af::array& cost) {
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
    const af::array& predBoxes, 
    const af::array& predLogits,
    const af::array& targetBoxes, 
    const af::array& targetClasses) const {

  af_print(targetClasses);
  af_print(targetBoxes);
  auto mask = targetClasses >= 0;
  af_print(targetBoxes(af::span, mask));

  // Create an M X N cost matrix where M is the number of targets and N is the number of preds
  
  // Class cost
  auto outProbs = softmax(fl::input(predLogits), 0).array();
  auto cost_class = 1 - outProbs(targetClasses, af::span);

  // Generalized IOU loss
  auto cost_giou =  -dataset::generalized_box_iou(targetBoxes, predBoxes);

  // Bbox Cost
  auto cost_bbox = dataset::cartesian(targetBoxes, predBoxes,
      [](const af::array& x, const af::array& y) {
        return af::sum(af::abs(x - y));
    });
  cost_bbox = dataset::flatten(cost_bbox, 0, 1);


  auto cost = cost_bbox_ * cost_bbox + cost_class_ * cost_class + cost_giou_ * cost_giou;
  return ::hungarian(cost);


}

std::vector<std::pair<af::array, af::array>> HungarianMatcher::forward(
    const af::array& predBoxes,
    const af::array& predLogits,
    const af::array& targetBoxes,
    const af::array& targetClasses) const {
  std::vector<std::pair<af::array, af::array>> results;
  for(int b = 0; b < predBoxes.dims(2); b++) {
    auto result = matchBatch(
      predBoxes(af::span, af::span, b),
      predLogits(af::span, af::span, b),
      targetBoxes(af::span, af::span, b),
      targetClasses(af::span, af::span, b));
    results.emplace_back(result);
  }
  return results;
};

} // end namespace cv
} // end namespace flashlight


