/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/criterion/SetCriterion.h"

#include <algorithm>
#include <cassert>
#include <numeric>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/distributed/DistributedApi.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/pkg/vision/dataset/BoxUtils.h"

namespace {

using namespace fl;

Tensor span(const Shape& inDims, const int index) {
  Shape dims(std::vector<Dim>(std::max(inDims.ndim(), index + 1), 1));
  if (index > inDims.ndim() - 1) {
    dims[index] = 1;
  } else {
    dims[index] = inDims[index];
  }
  return fl::iota(dims);
}

Shape calcStrides(const Shape& dims) {
  return {1, dims[0], dims[0] * dims[1], dims[0] * dims[1] * dims[2]};
};

Shape calcOutDims(const std::vector<Tensor>& coords) {
  unsigned maxNdim = 0;
  for (const auto& coord : coords) {
    if (coord.ndim() > maxNdim) {
      maxNdim = coord.ndim();
    }
  }

  Shape oDims(std::vector<Dim>(maxNdim, 1));

  for (const auto& coord : coords) {
    auto iDims = coord.shape();
    for (int i = 0; i < coord.ndim(); i++) {
      if (iDims[i] > 1 && oDims[i] == 1) {
        oDims[i] = iDims[i];
      }
      assert(iDims[i] == 1 || iDims[i] == oDims[i]);
    }
  }
  return oDims;
}

Tensor applyStrides(const std::vector<Tensor>& coords, const Shape& strides) {
  auto oDims = coords[0].shape();
  return std::inner_product(
      coords.begin(),
      coords.end(),
      strides.get().begin(),
      fl::full(oDims, 0),
      [](const Tensor& x, const Tensor& y) { return x + y; },
      [](const Tensor& x, int y) { return x * y; });
}

std::vector<Tensor> spanIfEmpty(const std::vector<Tensor>& coords, Shape dims) {
  std::vector<Tensor> result(coords.size());
  for (int i = 0; i < coords.size(); i++) {
    result[i] = (coords[i].isEmpty()) ? span(dims, i) : coords[i];
  }
  return result;
}

// Then, broadcast the indices
std::vector<Tensor> broadcastCoords(const std::vector<Tensor>& input) {
  std::vector<Tensor> result(input.size());
  auto oDims = calcOutDims(input);
  std::transform(
      input.begin(), input.end(), result.begin(), [&oDims](const Tensor& idx) {
        return detail::tileAs(idx, oDims);
      });
  return result;
}

Tensor ravelIndices(
    const std::vector<Tensor>& input_coords,
    const Shape& in_dims) {
  std::vector<Tensor> coords;
  coords = spanIfEmpty(input_coords, in_dims);
  coords = broadcastCoords(coords);
  return applyStrides(coords, calcStrides(in_dims));
}

Tensor index(const Tensor& in, const std::vector<Tensor>& idxs) {
  auto linearIndices = ravelIndices(idxs, in.shape());
  Tensor output = fl::full(linearIndices.shape(), 0., in.type());
  output.flat(fl::range(static_cast<long long>(linearIndices.elements()))) =
      in.flatten()(linearIndices);
  return output;
}

fl::Variable index(const fl::Variable& in, std::vector<Tensor> idxs) {
  auto idims = in.shape();
  auto result = index(in.tensor(), idxs);
  auto gradFunction = [idxs, idims](
                          std::vector<Variable>& inputs,
                          const Variable& grad_output) {
    if (!inputs[0].isGradAvailable()) {
      auto grad = fl::full(idims, 0., inputs[0].type());
      inputs[0].addGrad(Variable(grad, false));
      return;
    }
    auto grad = fl::Variable(fl::full(idims, 0, inputs[0].type()), false);
    auto linearIndices = ravelIndices(idxs, idims);
    grad.tensor()(linearIndices) = grad_output.tensor()(
        fl::range(static_cast<long long>(linearIndices.elements())));
    // TODO Can parallize this if needed but does not work for duplicate keys
    // for(int i = 0; i < linearIndices.elements(); i++) {
    // Tensor index = linearIndices(i);
    // grad.tensor()(index) += grad_output.tensor()(i);
    //}
    inputs[0].addGrad(grad);
  };
  return fl::Variable(result, {in.withoutData()}, gradFunction);
}

} // namespace

namespace fl::pkg::vision {

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

  for (int i = 0; i < predBoxesAux.dim(3); i++) {
    auto predBoxes = predBoxesAux(fl::span, fl::span, fl::span, i);
    auto predLogits =
        predLogitsAux(fl::span, fl::span, fl::span, fl::range(i, i + 1));

    std::vector<Tensor> targetBoxesArray(targetBoxes.size());
    std::vector<Tensor> targetClassesArray(targetClasses.size());
    std::transform(
        targetBoxes.begin(),
        targetBoxes.end(),
        targetBoxesArray.begin(),
        [](const Variable& in) { return in.tensor(); });
    std::transform(
        targetClasses.begin(),
        targetClasses.end(),
        targetClassesArray.begin(),
        [](const Variable& in) { return in.tensor(); });

    auto indices = matcher_.compute(
        predBoxes.tensor(),
        predLogits.tensor(),
        targetBoxesArray,
        targetClassesArray);

    int numBoxes = std::accumulate(
        targetBoxes.begin(),
        targetBoxes.end(),
        0,
        [](int curr, const Variable& label) { return curr + label.dim(1); });

    Tensor numBoxesArray = fl::fromScalar(numBoxes, fl::dtype::s32);
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
    const std::vector<std::pair<Tensor, Tensor>>& indices,
    const int numBoxes) {
  auto srcIdx = this->getSrcPermutationIdx(indices);
  if (srcIdx.first.isEmpty()) {
    return {
        {"lossGiou", fl::Variable(fl::fromScalar(0, predBoxes.type()), false)},
        {"lossBbox", fl::Variable(fl::fromScalar(0, predBoxes.type()), false)}};
  }
  auto colIdxs = fl::reshape(srcIdx.second, {1, srcIdx.second.dim(0)});
  auto batchIdxs = fl::reshape(srcIdx.first, {1, srcIdx.first.dim(0)});

  auto srcBoxes = index(predBoxes, {Tensor(), colIdxs, batchIdxs});

  int i = 0;
  std::vector<Variable> permuted;
  for (const auto& idx : indices) {
    auto targetIdxs = idx.first;
    auto reordered = targetBoxes[i](fl::span, targetIdxs);
    if (!reordered.isEmpty()) {
      permuted.emplace_back(reordered);
    }
    i += 1;
  }
  auto tgtBoxes = fl::concatenate(permuted, 1);

  auto costGiou =
      generalizedBoxIou(cxcywh2xyxy(srcBoxes), cxcywh2xyxy(tgtBoxes));

  // Extract diagonal
  auto dims = costGiou.shape();
  auto rng = fl::arange({dims[0]});
  costGiou = 1 - index(costGiou, {rng, rng, Tensor(), Tensor()});

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
    const std::vector<std::pair<Tensor, Tensor>>& indices,
    const int /*numBoxes*/) {
  assert(predLogits.dim(0) == numClasses_ + 1);

  auto target_classes_full = fl::full(
      // TODO: this thing requires predLogits to have > 2 dimensions
      {predLogits.dim(1), predLogits.dim(2), 1},
      numClasses_,
      predLogits.type());

  int i = 0;
  for (const auto& idx : indices) {
    auto targetIdxs = idx.first;
    auto srcIdxs = idx.second;
    auto reordered = targetClasses[i](targetIdxs);
    target_classes_full(srcIdxs, i) =
        fl::reshape(
            targetClasses[i].tensor()(targetIdxs),
            {static_cast<long long>(srcIdxs.elements()), 1})
            .astype(target_classes_full.type());
    i += 1;
  }

  auto softmaxed = logSoftmax(predLogits, 0);
  auto weight = fl::full({numClasses_ + 1}, 1.0f);
  weight.flat(numClasses_) = eosCoef_;
  auto weightVar = Variable(weight, false);
  auto lossCe = weightedCategoricalCrossEntropy(
      softmaxed,
      fl::Variable(target_classes_full.astype(fl::dtype::f32), false),
      weightVar,
      -1);
  return {{"lossCe", lossCe.astype(predLogits.type())}};
}

std::unordered_map<std::string, float> SetCriterion::getWeightDict() {
  return weightDict_;
}

std::pair<Tensor, Tensor> SetCriterion::getTgtPermutationIdx(
    const std::vector<std::pair<Tensor, Tensor>>& indices) {
  long batchSize = static_cast<long>(indices.size());
  auto batchIdxs = fl::full({1, 1, 1, batchSize}, -1);
  auto first = indices[0].first;
  auto dims = first.shape();
  auto tgtIdxs = fl::full({1, dims[0], batchSize}, -1);
  int idx = 0;
  for (const auto& pair : indices) {
    batchIdxs(0, 0, 0, idx) = fl::fromScalar(idx);
    tgtIdxs(fl::span, fl::span, idx) = pair.first;
    idx++;
  }
  return std::make_pair(batchIdxs, tgtIdxs);
}

std::pair<Tensor, Tensor> SetCriterion::getSrcPermutationIdx(
    const std::vector<std::pair<Tensor, Tensor>>& indices) {
  std::vector<fl::Variable> srcIdxs;
  std::vector<fl::Variable> batchIdxs;
  for (int i = 0; i < indices.size(); i++) {
    auto index = indices[i].second;
    if (!index.isEmpty()) {
      srcIdxs.emplace_back(index, false);
      auto batchIdx = fl::full(index.shape(), i, fl::dtype::s32);
      batchIdxs.emplace_back(batchIdx, false);
    }
  }
  fl::Variable srcIdx, batchIdx;
  if (!srcIdxs.empty()) {
    srcIdx = concatenate(srcIdxs, 0);
    batchIdx = concatenate(batchIdxs, 0);
  }
  return {batchIdx.tensor(), srcIdx.tensor()};
}

} // namespace fl
