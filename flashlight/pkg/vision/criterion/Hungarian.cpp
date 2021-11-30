/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "flashlight/pkg/vision/criterion/Hungarian.h"

#include "flashlight/lib/set/Hungarian.h"
#include "flashlight/pkg/vision/dataset/BoxUtils.h"

#include "flashlight/fl/autograd/Functions.h"

namespace {
using namespace fl;

Tensor softmax(const Tensor& input, const int dim) {
  auto maxvals = af::max(input, dim);
  af::dim4 tiledims(1, 1, 1, 1);
  tiledims[dim] = input.dims(dim);

  auto expInput = af::exp(input - af::tile(maxvals, tiledims));
  auto result = expInput / af::tile(af::sum(expInput, dim), tiledims);
  return result;
}

std::pair<Tensor, Tensor> hungarian(Tensor& cost) {
  cost = cost.T();
  const int M = cost.dim(0);
  const int N = cost.dim(1);
  std::vector<float> costHost(cost.size());
  std::vector<int> rowIdxs(M);
  std::vector<int> colIdxs(M);
  cost.host(costHost.data());
  fl::lib::set::hungarian(
      costHost.data(), rowIdxs.data(), colIdxs.data(), M, N);
  auto rowIdxsArray = Tensor::fromVector(rowIdxs);
  auto colIdxsArray = Tensor::fromVector(colIdxs);
  return {rowIdxsArray, colIdxsArray};
}
} // namespace

namespace fl {
namespace pkg {
namespace vision {

HungarianMatcher::HungarianMatcher(
    const float costClass,
    const float costBbox,
    const float costGiou)
    : costClass_(costClass), costBbox_(costBbox), costGiou_(costGiou){};

std::pair<Tensor, Tensor> HungarianMatcher::matchBatch(
    const Tensor& predBoxes,
    const Tensor& predLogits,
    const Tensor& targetBoxes,
    const Tensor& targetClasses) const {
  // Kind of a hack...
  if (targetClasses.isEmpty()) {
    return {fl::fromScalar(0), fl::fromScalar(0)};
  }

  // Create an M X N cost matrix where M is the number of targets and N is the
  // number of preds
  // Class cost
  auto outProbs = ::softmax(predLogits, 0);
  auto costClass = transpose((0 - outProbs(targetClasses, af::span)));

  // Generalized IOU loss
  Tensor costGiou =
      0 - generalizedBoxIou(cxcywh2xyxy(predBoxes), cxcywh2xyxy(targetBoxes));

  // Bbox Cost
  Tensor costBbox =
      cartesian(predBoxes, targetBoxes, [](const Tensor& x, const Tensor& y) {
        return fl::sum(fl::abs(x - y), {0});
      });
  costBbox = flatten(costBbox, 0, 1);

  auto cost =
      costBbox_ * costBbox + costClass_ * costClass + costGiou_ * costGiou;
  return ::hungarian(cost);
}

std::vector<std::pair<Tensor, Tensor>> HungarianMatcher::compute(
    const Tensor& predBoxes,
    const Tensor& predLogits,
    const std::vector<Tensor>& targetBoxes,
    const std::vector<Tensor>& targetClasses) const {
  std::vector<std::pair<Tensor, Tensor>> results;
  for (int b = 0; b < predBoxes.dims(2); b++) {
    auto result = matchBatch(
        predBoxes(af::span, af::span, b),
        predLogits(af::span, af::span, b),
        targetBoxes[b],
        targetClasses[b]);
    results.emplace_back(result);
  }
  return results;
};

} // namespace vision
} // namespace pkg
} // namespace fl
