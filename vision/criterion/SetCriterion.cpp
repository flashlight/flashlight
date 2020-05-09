#include "SetCriterion.h"
#include <iostream>

#include "vision/dataset/BoxUtils.h"

namespace {

using namespace fl;

af::array span(const af::dim4 inDims, const int index) {
  af::dim4 dims = {1, 1, 1, 1};
  dims[index] = inDims[index];
  return af::iota(dims);
}

af::dim4 strides(const af::dim4 dims) {
  return { 1, dims[0], dims[0] * dims[1], dims[0] * dims[1] * dims[2] };
}

af::array computeLinearIndex(
    af::array idx0,
    af::array idx1,
    af::array idx2,
    af::array idx3,
    const af::dim4 dims) {
    idx0 = (idx0.isempty()) ? span(dims, 0) : idx0;
    idx1 = (idx1.isempty()) ? span(dims, 1) : idx1;
    idx2 = (idx2.isempty()) ? span(dims, 2) : idx2;
    idx3 = (idx3.isempty()) ? span(dims, 3) : idx3;
    af::dim4 stride = strides(dims);
    af::array linearIndices = batchFunc(idx0 * stride[0], idx1 * stride[1], af::operator+);
    linearIndices = batchFunc(linearIndices, idx2 * stride[2], af::operator+);
    linearIndices = batchFunc(linearIndices, idx3 * stride[3], af::operator+);
    return linearIndices;
}


af::array lookup(
    const af::array& in,
    af::array idx0,
    af::array idx1,
    af::array idx2,
    af::array idx3) {
  auto linearIndices = computeLinearIndex(idx0, idx1, idx2, idx3, in.dims());
  af::array output = af::constant(0.0, linearIndices.dims());
  output(af::seq(linearIndices.elements())) = in(linearIndices);
  return output;
}

fl::Variable lookup(
    const fl::Variable& in,
    af::array idx0,
    af::array idx1,
    af::array idx2,
    af::array idx3) {
  auto idims = in.dims();
  auto result = lookup(in.array(), idx0, idx1, idx2, idx3);
  auto gradFunction = [idx0, idx1, idx2, idx3, idims](std::vector<Variable>& inputs,
                                              const Variable& grad_output) {
        af_print(grad_output.array());
        if (!inputs[0].isGradAvailable()) {
          auto grad = af::constant(0.0, idims);
          inputs[0].addGrad(Variable(grad, false));
        }
        auto grad = fl::Variable(af::constant(0, idims), false);
        auto linearIndices = computeLinearIndex(idx0, idx1, idx2, idx3, idims);
        // TODO Can parallize this if needed but does not work for duplicate keys
        for(int i = 0; i < linearIndices.elements(); i++) {
          af::array index = linearIndices(i);
          grad.array()(index) += grad_output.array()(i);
        }
        inputs[0].addGrad(grad);
  };
  return fl::Variable(result, { in.withoutData() }, gradFunction);
}
}

namespace fl {
namespace cv {

SetCriterion::SetCriterion(
    const int num_classes,
    const HungarianMatcher& matcher,
    const af::array& weight_dict,
    const float eos_coef,
    LossDict losses) :
  num_classes_(num_classes),
  matcher_(matcher),
  weight_dict_(weight_dict),
  eos_coef_(eos_coef),
  losses_(losses) { };

SetCriterion::LossDict SetCriterion::forward(
    const Variable& predBoxes,
    const Variable& predLogits,
    const Variable& targetBoxes,
    const Variable& targetClasses) {

      auto indices = matcher_.forward(predBoxes.array(),
          predLogits.array(), targetBoxes.array(), targetClasses.array());

      // TODO get number of boxes
      int numBoxes = 10;
      LossDict losses;

      auto labelLoss = lossLabels(predBoxes, predLogits, targetBoxes, targetClasses, indices, numBoxes);
      auto bboxLoss = lossBoxes(predBoxes, predLogits, targetBoxes, targetClasses, indices, numBoxes);
      losses.insert(labelLoss.begin(), labelLoss.end());
      losses.insert(bboxLoss.begin(), bboxLoss.end());
      return losses;
}

SetCriterion::LossDict SetCriterion::lossBoxes(
    const Variable& predBoxes,
    const Variable& predLogits,
    const Variable& targetBoxes,
    const Variable& targetClasses,
    const std::vector<std::pair<af::array, af::array>>& indices,
    const int numBoxes) {

  auto srcIdx = this->getSrcPermutationIdx(indices);
  auto tgtIdx = this->getTgtPermutationIdx(indices);
  auto srcBoxes = lookup(predBoxes, af::array(), srcIdx.second, af::array(), af::array());
  auto tgtBoxes = lookup(targetBoxes, af::array(), tgtIdx.second, af::array(), af::array());

  auto cost_giou =  0 - dataset::generalized_box_iou(srcBoxes, tgtBoxes);
  auto dims = cost_giou.dims();
  // Extract diagnal
  auto rng = af::range(dims[0]);
  cost_giou = lookup(cost_giou, rng, rng, af::array(), af::array());
  //cost_giou = sum(cost_giou) / numBoxes;

  auto loss_bbox = cv::dataset::l1_loss(srcBoxes, tgtBoxes);
  //loss_bbox = sum(loss_bbox) / numBoxes;

  return { {"loss_giou", cost_giou}, {"loss_bbox", loss_bbox }};
}

SetCriterion::LossDict SetCriterion::lossLabels(
    const Variable& predBoxes,
    const Variable& predLogits,
    const Variable& targetBoxes,
    const Variable& targetClasses,
    const std::vector<std::pair<af::array, af::array>>& indices,
    const int numBoxes) {

  auto srcIdx = this->getSrcPermutationIdx(indices);
  auto tgtIdx = this->getTgtPermutationIdx(indices);

  auto srcLogits = lookup(predLogits, af::array(), srcIdx.second, af::array(), af::array());
  auto tgtClasses = lookup(targetClasses, af::array(), tgtIdx.second, af::array(), af::array());

  af_print(srcLogits.array());
  af_print(tgtClasses.array());

  af_print(srcLogits.array());
  af_print(tgtClasses.array());
  auto tgtDims = tgtClasses.dims();
  tgtClasses =  moddims(tgtClasses, { tgtDims[1], tgtDims[2], tgtDims[3] });
  af_print(tgtClasses.array());
  auto loss_ce = categoricalCrossEntropy(logSoftmax(srcLogits, 0), tgtClasses);
  af_print(loss_ce.array())
  return { {"loss_ce", loss_ce} };
}

// TODO we can push all of this into HungarianMatcher. Do after testing.
std::pair<af::array, af::array> SetCriterion::getTgtPermutationIdx(
    const std::vector<std::pair<af::array, af::array>>& indices) {
  long batchSize = static_cast<long>(indices.size());
  auto batchIdxs = af::constant(-1, {1, 1, 1, batchSize});
  auto first = indices[0].first;
  auto dims = first.dims();
  auto tgtIdxs = af::constant(-1, {1, dims[0], batchSize});
  int idx = 0;
  for(auto pair : indices) {
    batchIdxs(0, 0, 0, idx) = af::constant(idx, { 1 });
    tgtIdxs(af::span, af::span, idx) = pair.first;
    idx++;
  }
  return std::make_pair(batchIdxs, tgtIdxs);
}


std::pair<af::array, af::array> SetCriterion::getSrcPermutationIdx(
    const std::vector<std::pair<af::array, af::array>>& indices) {
  long batchSize = static_cast<long>(indices.size());
  auto batchIdxs = af::constant(-1, {1, 1, batchSize});
  auto first = indices[0].first;
  auto dims = first.dims();
  auto srcIdxs = af::constant(-1, {1, dims[0], batchSize});
  int idx = 0;
  for(auto pair : indices) {
    af_print(pair.second);
    batchIdxs(0, 0, idx) = af::constant(idx, { 1 });
    srcIdxs(af::span, af::span, idx) = pair.second;
    idx++;
  }
  return std::make_pair(batchIdxs, srcIdxs);
}

}// end namespace cv
} // end namespace fl
