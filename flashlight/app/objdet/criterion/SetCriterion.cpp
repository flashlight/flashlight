#include "SetCriterion.h"

#include <assert.h>
#include <iostream>
#include <numeric>
#include <algorithm>

#include <af/array.h>

#include "flashlight/app/objdet/dataset/BoxUtils.h"
#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/Defines.h"

namespace {

using namespace fl;

Variable weightedCategoricalCrossEntropy(
    const Variable& input,
    const Variable& targets,
    const Variable& weight,
    ReduceMode reduction /* =ReduceMode::MEAN */,
    int ignoreIndex /* = -1 */) {
  // input -- [C, X1, X2, X3]
  // target -- [X1, X2, X3, 1]
  for (int i = 1; i < 4; i++) {
    if (input.dims(i) != targets.dims(i - 1)) {
      throw std::invalid_argument(
          "dimension mismatch in categorical cross entropy");
    }
  }
  if (targets.dims(3) != 1) {
    throw std::invalid_argument(
        "dimension mismatch in categorical cross entropy");
  }

  int C = input.dims(0);
  int X = targets.elements();
  if (af::anyTrue<bool>((targets.array() < 0) || (targets.array() >= C))) {
    throw std::invalid_argument(
        "target contains elements out of valid range [0, num_categories) "
        "in categorical cross entropy");
  }

  auto x = af::moddims(input.array(), af::dim4(C, X));
  auto y = af::moddims(targets.array(), af::dim4(1, X));

  auto A = af::range(af::dim4(C, X));
  auto B = af::tile(y, af::dim4(C));
  auto mask = -(A == B); // [C X]

  auto result = mask * x;
  result = af::batchFunc(result, weight.array(), af::operator*);
  auto ignoreMask = (y != ignoreIndex).as(s32); // [1 X]
  result = ignoreMask * af::sum(result, 0); // [1 X]

  Variable denominator;
  if (reduction == ReduceMode::NONE) {
    result = af::moddims(result, targets.dims()); // [X1 X2 X3]
  } else if (reduction == ReduceMode::MEAN) {
    denominator = Variable(af::sum(ignoreMask, 1), false);
    result = af::sum(result, 1) / denominator.array(); // [1]
  } else if (reduction == ReduceMode::SUM) {
    result = af::sum(result, 1); // [1]
  } else {
    throw std::invalid_argument(
        "unknown reduction method for categorical cross entropy");
  }

  auto inputDims = input.dims();
  auto gradFunc = [C, X, mask, ignoreMask, denominator, reduction, inputDims](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto grad = gradOutput.array();
    if (reduction == ReduceMode::NONE) {
      grad = af::moddims(grad, af::dim4(1, X));
    } else if (reduction == ReduceMode::MEAN) {
      grad = af::tile(grad / denominator.array(), af::dim4(1, X));
    } else if (reduction == ReduceMode::SUM) {
      grad = af::tile(grad, af::dim4(1, X));
    }
    // [1 X]
    auto weightArray = inputs[2].array();
    grad *= ignoreMask;
    grad = af::tile(grad, af::dim4(C)) * mask;
    grad = af::moddims(grad, inputDims);
    grad = af::batchFunc(grad, weightArray, af::operator*);;
    inputs[0].addGrad(Variable(af::moddims(grad, inputDims), false));
  };

  return Variable(result, {input.withoutData(), targets, weight}, gradFunc);
}


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
        grad.array()(linearIndices) = 
          grad_output.array()(af::seq(linearIndices.elements()));
        // TODO Can parallize this if needed but does not work for duplicate keys
        //for(int i = 0; i < linearIndices.elements(); i++) {
          //af::array index = linearIndices(i);
          //grad.array()(index) += grad_output.array()(i);
        //}
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
namespace app {
namespace objdet {

SetCriterion::SetCriterion(
    const int num_classes,
    const HungarianMatcher& matcher,
    const std::unordered_map<std::string, float> weight_dict,
    const float eos_coef,
    LossDict losses) :
  num_classes_(num_classes),
  matcher_(matcher),
  weight_dict_(weight_dict),
  eos_coef_(eos_coef),
  losses_(losses) { };

SetCriterion::LossDict SetCriterion::forward(
    const Variable& predBoxesAux,
    const Variable& predLogitsAux,
    const std::vector<Variable>& targetBoxes,
    const std::vector<Variable>& targetClasses) {

    LossDict losses;

    for(int i = 0; i < predBoxesAux.dims(3); i++) {

      auto predBoxes = predBoxesAux(af::span, af::span, af::span, af::seq(i, i));
      auto predLogits = predLogitsAux(af::span, af::span, af::span, af::seq(i, i));

      auto indices = matcher_.forward(predBoxes, predLogits, targetBoxes, targetClasses);

      int numBoxes = std::accumulate(targetBoxes.begin(), targetBoxes.end(), 0,
          [](int curr, const Variable& label) { return curr + label.dims(1);  }
      );
      // TODO clamp number of boxes based on world size
      // https://github.com/fairinternal/detection-transformer/blob/master/models/detr.py#L168


      auto labelLoss = lossLabels(predBoxes, predLogits, targetBoxes, targetClasses, indices, numBoxes);
      auto bboxLoss = lossBoxes(predBoxes, predLogits, targetBoxes, targetClasses, indices, numBoxes);
      for(std::pair<std::string, Variable> l : labelLoss) {
        losses[l.first + "_" + std::to_string(i)] = l.second;
      }
      for(std::pair<std::string, Variable> l : bboxLoss) {
        losses[l.first + "_" + std::to_string(i)] = l.second;
      }
    }
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
  if (srcIdx.first.isempty()) {
    return { { "loss_giou", fl::Variable(af::constant(0, {1}), false) } , 
              {"loss_bbox", fl::Variable(af::constant(0, {1}), false) }};
  }
  auto colIdxs = af::moddims(srcIdx.second, { 1, srcIdx.second.dims(0) });
  auto batchIdxs = af::moddims(srcIdx.first, {1, srcIdx.first.dims(0) });
  auto srcBoxes = index(predBoxes, { af::array(), colIdxs, batchIdxs, af::array() });


  int i = 0;
  std::vector<Variable> permuted;
  for(auto idx: indices) {
    auto targetIdxs = idx.first;
    auto reordered = targetBoxes[i](af::span, targetIdxs);
    if(!reordered.isempty()) {
      permuted.emplace_back(reordered);
    }
    i += 1;
  }
  auto tgtBoxes = fl::concatenate(permuted, 1);


  auto cost_giou =  generalized_box_iou(
      cxcywh_to_xyxy(srcBoxes), 
      cxcywh_to_xyxy(tgtBoxes)
  );

  // Extract diagnal
  auto dims = cost_giou.dims();
  auto rng = af::range(dims[0]);
  cost_giou = 1 - index(cost_giou, { rng, rng, af::array(), af::array() });
  cost_giou = sum(cost_giou, { 0 } ) / numBoxes;

  auto loss_bbox = l1_loss(srcBoxes, tgtBoxes);
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
  int num_classes = softmaxed.dims(0);
  auto weight = af::constant(1, num_classes);
  weight(num_classes - 1) = eos_coef_;
  auto weightVar = Variable(weight, false);
  auto loss_ce = weightedCategoricalCrossEntropy(
      softmaxed, fl::Variable(target_classes_full, false), weightVar, ReduceMode::MEAN, -1);
  return { {"loss_ce", loss_ce} };
}

std::unordered_map<std::string, float> SetCriterion::getWeightDict() {
  return weight_dict_;
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

  //std::vector<fl::Variable> srcIdxs(indices.size());
  std::vector<fl::Variable> srcIdxs;
  std::vector<fl::Variable> batchIdxs;
  //std::transform(indices.begin(), indices.end(), srcIdxs.begin(),
      //[](std::pair<af::array, af::array> idxs) { return Variable(idxs.second, false); } 
  //);
  for(int i = 0; i < indices.size(); i++) {
    auto index = indices[i].second;
    if(!index.isempty()) {
      srcIdxs.emplace_back(Variable(index, false));
      auto batchIdx = af::constant(i, index.dims(), s32);
      batchIdxs.emplace_back(Variable(batchIdx, false));
    }
  }
  fl::Variable srcIdx, batchIdx;
  if (srcIdxs.size() > 0) {
    srcIdx = concatenate(srcIdxs, 0);
    batchIdx = concatenate(batchIdxs, 0);
  }
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

} // namespace objdet
} // namespace app
} // namespace fl
