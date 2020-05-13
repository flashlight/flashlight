#include "SetCriterion.h"

#include <assert.h>
#include <iostream>
#include <numeric>
#include <algorithm>

#include <af/array.h>

#include "vision/dataset/BoxUtils.h"
#include "flashlight/autograd/autograd.h"

namespace {

using namespace fl;

af::array span(const af::dim4 inDims, const int index) {
  af::dim4 dims = {1, 1, 1, 1};
  dims[index] = inDims[index];
  return af::iota(dims);
}

af::dim4 calcStrides(const af::dim4 dims) {
  af::dim4 oDims;
  return { 1, dims[0], dims[0] * dims[1], dims[0] * dims[1] * dims[2] };
};

af::dim4 calcOutDims(
    const std::vector<af::array>& coords) {
    af::dim4 oDims = {1, 1, 1, 1};
    for(auto coord : coords) {
      auto iDims = coord.dims();
      for(int i = 0; i < 4; i++) {
        if(iDims[i] > 1 && oDims[i] == 1) {
          oDims[i] = iDims[i];
        }
        assert(iDims[i] == 1 || iDims[i] == oDims[i]);
      }
    }
    return oDims;
}

af::array applyStrides(
    const std::vector<af::array>& coords, 
    af::dim4 strides) {
  auto oDims = coords[0].dims();
  return std::inner_product(
        coords.begin(), coords.end(), strides.get(), af::constant(0, oDims),
        [](const af::array& x, const af::array y) { return x + y; },
        [](const af::array& x, int y) { return x * y; }
  );
}

std::vector<af::array> spanIfEmpty(
    const std::vector<af::array>& coords,
    af::dim4 dims) {
  std::vector<af::array> result(coords.size());
    for(int i = 0; i < 4; i++) {
      result[i] = (coords[i].isempty()) ? span(dims, i) : coords[i];
    }
    return result;
}

// Then, broadcast the indices
std::vector<af::array> broadcastCoords(
    const std::vector<af::array>& input
    ) {
    std::vector<af::array> result(input.size());
    auto oDims = calcOutDims(input);
    std::transform(input.begin(), input.end(), result.begin(), 
        [&oDims](const af::array& idx) 
        { return detail::tileAs(idx, oDims); }
    );
    return result;

}

af::array ravelIndices(
    const std::vector<af::array>& input_coords,
    const af::dim4 in_dims) {

  std::vector<af::array> coords;
  coords = spanIfEmpty(input_coords, in_dims); 
  coords = broadcastCoords(coords);
  return applyStrides(coords, calcStrides(in_dims));
}

af::array index(
    const af::array& in,
    const std::vector<af::array> idxs
    ) {
  auto linearIndices = ravelIndices(idxs, in.dims());
  af::array output = af::constant(0.0, linearIndices.dims());
  output(af::seq(linearIndices.elements())) = in(linearIndices);
  return output;
}


af::array index(
    const af::array& in,
    af::array idx,
    const int dim
) { 
  std::vector<af::array> idxs(4);
  idxs[dim] = idx;
  return index(in, idxs);
}




fl::Variable index(
    const fl::Variable& in,
    std::vector<af::array> idxs
 ) {
  auto idims = in.dims();
  auto result = index(in.array(), idxs);
  auto gradFunction = [idxs, idims](std::vector<Variable>& inputs,
                                              const Variable& grad_output) {
        af_print(grad_output.array());
        if (!inputs[0].isGradAvailable()) {
          auto grad = af::constant(0.0, idims);
          inputs[0].addGrad(Variable(grad, false));
        }
        auto grad = fl::Variable(af::constant(0, idims), false);
        auto linearIndices = ravelIndices(idxs, idims);
        // TODO Can parallize this if needed but does not work for duplicate keys
        for(int i = 0; i < linearIndices.elements(); i++) {
          af::array index = linearIndices(i);
          grad.array()(index) += grad_output.array()(i);
        }
        inputs[0].addGrad(grad);
  };
  return fl::Variable(result, { in.withoutData() }, gradFunction);
}

fl::Variable index(
    const fl::Variable& in,
    const af::array idx,
    const int dim) {
  std::vector<af::array> idxs(4);
  idxs[dim] = idx;
  return index(in, idxs);
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
    const std::vector<Variable>& targetBoxes,
    const std::vector<Variable>& targetClasses) {

      auto indices = matcher_.forward(predBoxes, predLogits, targetBoxes, targetClasses);

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
    const std::vector<Variable>& targetBoxes,
    const std::vector<Variable>& targetClasses,
    const std::vector<std::pair<af::array, af::array>>& indices,
    const int numBoxes) {

  auto srcIdx = this->getSrcPermutationIdx(indices);
  auto tgtIdx = this->getTgtPermutationIdx(indices);
  auto srcBoxes = index(predBoxes, srcIdx.second, 1);
  // TODO concat target boxes
  auto tgtBoxes = index(targetBoxes[0], tgtIdx.second, 1);

  auto cost_giou =  dataset::generalized_box_iou(
      dataset::cxcywh_to_xyxy(srcBoxes), 
      dataset::cxcywh_to_xyxy(tgtBoxes)
  );
  auto dims = cost_giou.dims();
  // Extract diagnal
  auto rng = af::range(dims[0]);
  cost_giou = 1 - index(cost_giou, { rng, rng, af::array(), af::array() });
  // TODO 
  //cost_giou = sum(cost_giou) / numBoxes;

  auto loss_bbox = cv::dataset::l1_loss(srcBoxes, tgtBoxes);
  // TODO
  //loss_bbox = sum(loss_bbox) / numBoxes;

  return { {"loss_giou", cost_giou}, {"loss_bbox", loss_bbox }};
}

SetCriterion::LossDict SetCriterion::lossLabels(
    const Variable& predBoxes,
    const Variable& predLogits,
    const std::vector<Variable>& targetBoxes,
    const std::vector<Variable>& targetClasses,
    const std::vector<std::pair<af::array, af::array>>& indices,
    const int numBoxes) {

  auto srcIdx = this->getSrcPermutationIdx(indices);
  auto tgtIdx = this->getTgtPermutationIdx(indices);

  auto srcLogits = index(predLogits, srcIdx.second, 1);
  auto tgtClasses = index(targetClasses[0],  tgtIdx.second, 1);

  auto tgtDims = tgtClasses.dims();
  tgtClasses =  moddims(tgtClasses, { tgtDims[1], tgtDims[2], tgtDims[3] });
  auto softmaxed = logSoftmax(srcLogits, 0)
  af_print(softmaxed.array());
  auto loss_ce = categoricalCrossEntropy(softmaxed, tgtClasses);
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
