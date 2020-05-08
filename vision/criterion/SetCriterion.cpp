#include "SetCriterion.h"
#include <iostream>

#include "vision/dataset/BoxUtils.h"

namespace {

using namespace fl;

// TODO make work for batching / dimension interpolation
// TODO make work with leaving out arguments
af::array lookup(
    const af::array& in,
    af::array idx0,
    af::array idx1,
    af::array idx2,
    af::array idx3) {
  auto inDims = in.dims();
  idx0 = (idx0.isempty()) ? af::iota({inDims[0]}) : idx0;
  idx1 = (idx1.isempty()) ? af::iota({1, inDims[1]}) : idx1;
  idx2 = (idx2.isempty()) ? af::iota({1, 1, inDims[2]}) : idx2;
  idx3 = (idx3.isempty()) ? af::iota({1, 1, 1,  inDims[3]}) : idx3;
  af::dim4 stride = { 1, inDims[0], inDims[0] * inDims[1], inDims[0] * inDims[1] * inDims[2] };
  af::array linearIndices = batchFunc(idx0 * stride[0], idx1 * stride[1], af::operator+);
  linearIndices = batchFunc(linearIndices, idx2 * stride[2], af::operator+);
  linearIndices = batchFunc(linearIndices, idx3 * stride[3], af::operator+);
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
        auto inDims = idims;
        auto idx0_ = (idx0.isempty()) ? af::iota({idims[0]}) : idx0;
        auto idx1_ = (idx1.isempty()) ? af::iota({1, idims[1]}) : idx1;
        auto idx2_ = (idx2.isempty()) ? af::iota({1, 1, idims[2]}) : idx2;
        auto idx3_ = (idx3.isempty()) ? af::iota({1, 1, 1,  idims[3]}) : idx3;
        af::dim4 stride = { 1, idims[0], idims[0] * idims[1], idims[0] * idims[1] * idims[2] };
        af::array linearIndices = batchFunc(idx0_ * stride[0], idx1_ * stride[1], af::operator+);
        linearIndices = batchFunc(linearIndices, idx2_ * stride[2], af::operator+);
        linearIndices = batchFunc(linearIndices, idx3_ * stride[3], af::operator+);
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

      return lossBoxes(predBoxes, predLogits, targetBoxes, targetClasses, indices, numBoxes);
}

SetCriterion::LossDict SetCriterion::lossBoxes(
    const Variable& predBoxes,
    const Variable& predLogits,
    const Variable& targetBoxes,
    const Variable& targetClasses,
    const std::vector<std::pair<af::array, af::array>>& indices,
    const int numBoxes) {
  auto srcIdx = this->getSrcPermutationIdx(indices);
  af_print(srcIdx.second);
  auto srcBoxes = lookup(predBoxes, af::array(), srcIdx.second, af::array(), af::array());
  auto tgtIdx = this->getTgtPermutationIdx(indices);
  auto tgtBoxes = lookup(targetBoxes, af::array(), tgtIdx.second, af::array(), af::array());
  af_print(tgtIdx.second);
  af_print(srcBoxes.array());
  af_print(tgtBoxes.array());
  auto cost_giou =  0 - dataset::generalized_box_iou(srcBoxes, tgtBoxes);
  auto dims = cost_giou.dims();
  af_print(cost_giou.array());
  cost_giou = lookup(cost_giou, af::range(dims[0]), af::range(dims[0]), af::array(), af::array());
  return { {"cost_giou", cost_giou} };
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
    tgtIdxs(af::span, af::span, idx)
      = af::moddims(pair.first, {1, dims[0], 1});
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
    srcIdxs(af::span, af::span, idx)
      = af::moddims(pair.second, {1, dims[0], 1});
    idx++;
  }
  std::cout << " Done " << std::endl;
  return std::make_pair(batchIdxs, srcIdxs);
}

}// end namespace cv
} // end namespace fl
