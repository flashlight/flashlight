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

void assign(
    af::array& out,
    const af::array& in,
    const std::vector<af::array> idxs
    ) {
  auto linearIndices = ravelIndices(idxs, in.dims());
  out(linearIndices) = in(af::seq(linearIndices.elements()));
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

      int numBoxes = std::accumulate(targetBoxes.begin(), targetBoxes.end(), 0,
          [](int curr, const Variable& label) { return curr + label.dims(1);  }
      );
      // TODO clamp number of boxes based on world size
      // https://github.com/fairinternal/detection-transformer/blob/master/models/detr.py#L168

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
  auto colIdxs = af::moddims(srcIdx.second, { 1, srcIdx.second.dims(0) });
  auto batchIdxs = af::moddims(srcIdx.first, {1, srcIdx.first.dims(0) });
  auto srcBoxes = index(predBoxes, { af::array(), colIdxs, batchIdxs, af::array() });


  int i = 0;
  std::vector<Variable> permuted(targetBoxes.size());
  for(auto idx: indices) {
    auto targetIdxs = idx.first;
    auto reordered = targetBoxes[i](af::span, targetIdxs);
    permuted[i] = reordered;
    i += 1;
  }
  auto tgtBoxes = fl::concatenate(permuted, 1);


  auto cost_giou =  dataset::generalized_box_iou(
      dataset::cxcywh_to_xyxy(srcBoxes), 
      dataset::cxcywh_to_xyxy(tgtBoxes)
  );

  // Extract diagnal
  auto dims = cost_giou.dims();
  auto rng = af::range(dims[0]);
  cost_giou = 1 - index(cost_giou, { rng, rng, af::array(), af::array() });
  cost_giou = sum(cost_giou, { 0 } ) / numBoxes;

  auto loss_bbox = cv::dataset::l1_loss(srcBoxes, tgtBoxes);
  loss_bbox = sum(loss_bbox, { 0 } ) / numBoxes;

  return { {"loss_giou", cost_giou}, {"loss_bbox", loss_bbox }};
}

SetCriterion::LossDict SetCriterion::lossLabels(
    const Variable& predBoxes,
    const Variable& predLogits,
    const std::vector<Variable>& targetBoxes,
    const std::vector<Variable>& targetClasses,
    const std::vector<std::pair<af::array, af::array>>& indices,
    const int numBoxes) {

  auto target_classes_full = af::constant(predLogits.dims(0) - 1, 
      { predLogits.dims(1), predLogits.dims(2), predLogits.dims(3) } , f32);

  int i = 0;
  for(auto idx: indices) {
    auto targetIdxs = idx.first;
    auto srcIdxs = idx.second;
    auto reordered = targetClasses[i](targetIdxs);
    target_classes_full(srcIdxs, i) = targetClasses[i].array()(targetIdxs);
    i += 1;
  }

  auto softmaxed = logSoftmax(predLogits, 0);
  auto loss_ce = categoricalCrossEntropy(
      softmaxed, fl::Variable(target_classes_full, false));
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

  std::vector<fl::Variable> srcIdxs(indices.size());
  std::transform(indices.begin(), indices.end(), srcIdxs.begin(),
      [](std::pair<af::array, af::array> idxs) { return Variable(idxs.second, false); } 
  );

  std::vector<fl::Variable> batchIdxs(indices.size());
  for(uint64_t i = 0; i < indices.size(); i ++) {
    auto result = af::constant(i, indices[i].second.dims(), s32);
    batchIdxs[i] = Variable(result, false);;
  };

  auto srcIdx = concatenate(srcIdxs, 0);
  auto batchIdx = concatenate(batchIdxs, 0);
  return { batchIdx.array(), srcIdx.array() };
  //long batchSize = static_cast<long>(indices.size());
  //auto batchIdxs = af::constant(-1, {1, 1, batchSize});
  //auto first = indices[0].first;
  //auto dims = first.dims();
  //auto srcIdxs = af::constant(-1, {1, dims[0], batchSize});
  //int idx = 0;
  //for(auto pair : indices) {
    //batchIdxs(0, 0, idx) = af::constant(idx, { 1 });
    //srcIdxs(af::span, af::span, idx) = pair.second;
    //idx++;
  //}
  //return std::make_pair(batchIdxs, srcIdxs);
}

}// end namespace cv
} // end namespace fl
