#include "Hungarian.h"
#include "vision/dataset/BoxUtils.h"
#include "vision/criterion/HungarianLib.h"

#include "flashlight/nn/nn.h"

namespace {

using namespace fl::cv;

float bbox_loss(const af::array& input, const af::array& target) {
  return af::sum((target - input)).scalar<float>();
}

std::pair<af::array, af::array> matchBatch(const af::array& predBoxes, const af::array& predLogits,
    const af::array& targetBoxes, const af::array& targetClasses) {
  const int num_preds = predBoxes.dims(1);
  const int num_targets = targetBoxes.dims(1);
  auto outProbs = softmax(fl::input(predLogits), 0).array();

  auto cost_giou =  -dataset::generalized_box_iou(targetBoxes, predBoxes);
  auto cost_giou_vec = std::vector<float>(cost_giou.elements());

  cost_giou.host(cost_giou_vec.data());
  std::vector<int> rowIdxs(num_targets);
  std::vector<int> colIdxs(num_targets);
  hungarian(cost_giou_vec.data(), rowIdxs.data(), colIdxs.data(), num_targets, num_preds);

  auto rowIdxsArray = af::array(num_targets, rowIdxs.data());
  auto colIdxsArray = af::array(num_targets, colIdxs.data());
  return { rowIdxsArray, colIdxsArray };

}

}

namespace fl {
namespace cv {

HungarianMatcher::HungarianMatcher(
    const float cost_class,
    const float cost_bbox,
    const float cost_giou) :
  cost_class_(cost_class), cost_bbox_(cost_bbox), cost_giou(cost_giou) {
};

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


