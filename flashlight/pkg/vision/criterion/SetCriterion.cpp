/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/criterion/SetCriterion.h"

#include <assert.h>
#include <algorithm>
#include <numeric>

#include <af/array.h>

#include "flashlight/pkg/vision/dataset/BoxUtils.h"
#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/distributed/DistributedApi.h"

namespace {

using namespace fl;

af::array span(const af::dim4& inDims, const int index) {
  af::dim4 dims = {1, 1, 1, 1};
  dims[index] = inDims[index];
  return af::iota(dims);
}

af::dim4 calcStrides(const af::dim4& dims) {
  af::dim4 oDims;
  return {1, dims[0], dims[0] * dims[1], dims[0] * dims[1] * dims[2]};
};

af::dim4 calcOutDims(const std::vector<af::array>& coords) {
  af::dim4 oDims = {1, 1, 1, 1};
  for (auto coord : coords) {
    auto iDims = coord.dims();
    for (int i = 0; i < 4; i++) {
      if (iDims[i] > 1 && oDims[i] == 1) {
        oDims[i] = iDims[i];
      }
      assert(iDims[i] == 1 || iDims[i] == oDims[i]);
    }
  }
  return oDims;
}

af::array applyStrides(const std::vector<af::array>& coords, af::dim4 strides) {
  auto oDims = coords[0].dims();
  return std::inner_product(
      coords.begin(),
      coords.end(),
      strides.get(),
      af::constant(0, oDims),
      [](const af::array& x, const af::array& y) { return x + y; },
      [](const af::array& x, int y) { return x * y; });
}

std::vector<af::array> spanIfEmpty(
    const std::vector<af::array>& coords,
    af::dim4 dims) {
  std::vector<af::array> result(coords.size());
  for (int i = 0; i < 4; i++) {
    result[i] = (coords[i].isempty()) ? span(dims, i) : coords[i];
  }
  return result;
}

// Then, broadcast the indices
std::vector<af::array> broadcastCoords(const std::vector<af::array>& input) {
  std::vector<af::array> result(input.size());
  auto oDims = calcOutDims(input);
  std::transform(
      input.begin(),
      input.end(),
      result.begin(),
      [&oDims](const af::array& idx) { return detail::tileAs(idx, oDims); });
  return result;
}

af::array ravelIndices(
    const std::vector<af::array>& input_coords,
    const af::dim4& in_dims) {
  std::vector<af::array> coords;
  coords = spanIfEmpty(input_coords, in_dims);
  coords = broadcastCoords(coords);
  return applyStrides(coords, calcStrides(in_dims));
}

af::array index(const af::array& in, const std::vector<af::array>& idxs) {
  auto linearIndices = ravelIndices(idxs, in.dims());
  af::array output = af::constant(0.0, linearIndices.dims(), in.type());
  output(af::seq(linearIndices.elements())) = in(linearIndices);
  return output;
}

fl::Variable index(const fl::Variable& in, std::vector<af::array> idxs) {
  auto idims = in.dims();
  auto result = index(in.array(), idxs);
  auto gradFunction = [idxs, idims](
                          std::vector<Variable>& inputs,
                          const Variable& grad_output) {
    if (!inputs[0].isGradAvailable()) {
      auto grad = af::constant(0.0, idims, inputs[0].type());
      inputs[0].addGrad(Variable(grad, false));
      return;
    }
    auto grad = fl::Variable(af::constant(0, idims, inputs[0].type()), false);
    auto linearIndices = ravelIndices(idxs, idims);
    grad.array()(linearIndices) =
        grad_output.array()(af::seq(linearIndices.elements()));
    // TODO Can parallize this if needed but does not work for duplicate keys
    // for(int i = 0; i < linearIndices.elements(); i++) {
    // af::array index = linearIndices(i);
    // grad.array()(index) += grad_output.array()(i);
    //}
    inputs[0].addGrad(grad);
  };
  return fl::Variable(result, {in.withoutData()}, gradFunction);
}

} // namespace

namespace fl {
namespace app {
namespace objdet {

SetCriterion::SetCriterion(
    const int numClasses,
    const HungarianMatcher& matcher,
    const std::unordered_map<std::string, float>& weightDict,
    const float eosCoef)
    : numClasses_(numClasses),
      matcher_(matcher),
      weightDict_(weightDict),
      eosCoef_(eosCoef){};

SetCriterion::LossDict SetCriterion::forward(
    const Variable& predBoxesAux,
    const Variable& predLogitsAux,
    const std::vector<Variable>& targetBoxes,
    const std::vector<Variable>& targetClasses) {
  LossDict losses;

  for (int i = 0; i < predBoxesAux.dims(3); i++) {
    auto predBoxes = predBoxesAux(af::span, af::span, af::span, af::seq(i, i));
    auto predLogits =
        predLogitsAux(af::span, af::span, af::span, af::seq(i, i));

    std::vector<af::array> targetBoxesArray(targetBoxes.size());
    std::vector<af::array> targetClassesArray(targetClasses.size());
    std::transform(
        targetBoxes.begin(),
        targetBoxes.end(),
        targetBoxesArray.begin(),
        [](const Variable& in) { return in.array(); });
    std::transform(
        targetClasses.begin(),
        targetClasses.end(),
        targetClassesArray.begin(),
        [](const Variable& in) { return in.array(); });

    auto indices = matcher_.compute(
        predBoxes.array(),
        predLogits.array(),
        targetBoxesArray,
        targetClassesArray);

    int numBoxes = std::accumulate(
        targetBoxes.begin(),
        targetBoxes.end(),
        0,
        [](int curr, const Variable& label) { return curr + label.dims(1); });

    af::array numBoxesArray = af::constant(numBoxes, 1, af::dtype::s32);
    if (isDistributedInit()) {
      allReduce(numBoxesArray);
    }
    numBoxes = numBoxesArray.scalar<int>();
    numBoxes = std::max(numBoxes / fl::getWorldSize(), 1);

    auto labelLoss = lossLabels(
        predBoxes, predLogits, targetBoxes, targetClasses, indices, numBoxes);
    auto bboxLoss = lossBoxes(
        predBoxes, predLogits, targetBoxes, targetClasses, indices, numBoxes);
    for (std::pair<std::string, Variable> l : labelLoss) {
      losses[l.first + "_" + std::to_string(i)] = l.second;
    }
    for (std::pair<std::string, Variable> l : bboxLoss) {
      losses[l.first + "_" + std::to_string(i)] = l.second;
    }
  }
  return losses;
}

SetCriterion::LossDict SetCriterion::lossBoxes(
    const Variable& predBoxes,
    const Variable& /*predLogits*/,
    const std::vector<Variable>& targetBoxes,
    const std::vector<Variable>& /*targetClasses*/,
    const std::vector<std::pair<af::array, af::array>>& indices,
    const int numBoxes) {
  auto srcIdx = this->getSrcPermutationIdx(indices);
  if (srcIdx.first.isempty()) {
    return {
        {"lossGiou",
         fl::Variable(af::constant(0, {1, 1, 1, 1}, predBoxes.type()), false)},
        {"lossBbox",
         fl::Variable(af::constant(0, {1, 1, 1, 1}, predBoxes.type()), false)}};
  }
  auto colIdxs = af::moddims(srcIdx.second, {1, srcIdx.second.dims(0)});
  auto batchIdxs = af::moddims(srcIdx.first, {1, srcIdx.first.dims(0)});
  auto srcBoxes =
      index(predBoxes, {af::array(), colIdxs, batchIdxs, af::array()});

  int i = 0;
  std::vector<Variable> permuted;
  for (auto idx : indices) {
    auto targetIdxs = idx.first;
    auto reordered = targetBoxes[i](af::span, targetIdxs);
    if (!reordered.isempty()) {
      permuted.emplace_back(reordered);
    }
    i += 1;
  }
  auto tgtBoxes = fl::concatenate(permuted, 1);

  auto costGiou =
      generalizedBoxIou(cxcywh2xyxy(srcBoxes), cxcywh2xyxy(tgtBoxes));

  // Extract diagnal
  auto dims = costGiou.dims();
  auto rng = af::range(dims[0]);
  costGiou = 1 - index(costGiou, {rng, rng, af::array(), af::array()});
  costGiou = sum(costGiou, {0}) / numBoxes;

  auto lossBbox = l1Loss(srcBoxes, tgtBoxes);
  lossBbox = sum(lossBbox, {0}) / numBoxes;

  return {{"lossGiou", costGiou}, {"lossBbox", lossBbox}};
}

SetCriterion::LossDict SetCriterion::lossLabels(
    const Variable& /*predBoxes*/,
    const Variable& predLogits,
    const std::vector<Variable>& /*targetBoxes*/,
    const std::vector<Variable>& targetClasses,
    const std::vector<std::pair<af::array, af::array>>& indices,
    const int /*numBoxes*/) {
  assert(predLogits.dims(0) == numClasses_ + 1);

  auto target_classes_full = af::constant(
      numClasses_,
      {predLogits.dims(1), predLogits.dims(2), predLogits.dims(3)},
      predLogits.type());

  int i = 0;
  for (auto idx : indices) {
    auto targetIdxs = idx.first;
    auto srcIdxs = idx.second;
    auto reordered = targetClasses[i](targetIdxs);
    target_classes_full(srcIdxs, i) = targetClasses[i].array()(targetIdxs);
    i += 1;
  }

  auto softmaxed = logSoftmax(predLogits, 0);
  auto weight = af::constant(1.0f, numClasses_ + 1);
  weight(numClasses_) = eosCoef_;
  auto weightVar = Variable(weight, false);
  auto lossCe = weightedCategoricalCrossEntropy(
      softmaxed, fl::Variable(target_classes_full, false), weightVar, -1);
  return {{"lossCe", lossCe.as(predLogits.type())}};
}

std::unordered_map<std::string, float> SetCriterion::getWeightDict() {
  return weightDict_;
}

std::pair<af::array, af::array> SetCriterion::getTgtPermutationIdx(
    const std::vector<std::pair<af::array, af::array>>& indices) {
  long batchSize = static_cast<long>(indices.size());
  auto batchIdxs = af::constant(-1, {1, 1, 1, batchSize});
  auto first = indices[0].first;
  auto dims = first.dims();
  auto tgtIdxs = af::constant(-1, {1, dims[0], batchSize});
  int idx = 0;
  for (auto pair : indices) {
    batchIdxs(0, 0, 0, idx) = af::constant(idx, {1, 1, 1, 1});
    tgtIdxs(af::span, af::span, idx) = pair.first;
    idx++;
  }
  return std::make_pair(batchIdxs, tgtIdxs);
}

std::pair<af::array, af::array> SetCriterion::getSrcPermutationIdx(
    const std::vector<std::pair<af::array, af::array>>& indices) {
  std::vector<fl::Variable> srcIdxs;
  std::vector<fl::Variable> batchIdxs;
  for (int i = 0; i < indices.size(); i++) {
    auto index = indices[i].second;
    if (!index.isempty()) {
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
  return {batchIdx.array(), srcIdx.array()};
}

} // namespace objdet
} // namespace app
} // namespace fl
