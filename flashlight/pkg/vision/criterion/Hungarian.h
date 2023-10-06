/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {
namespace pkg {
namespace vision {

class HungarianMatcher {
 public:
  HungarianMatcher() = default;

  HungarianMatcher(
      const float costClass,
      const float costBbox,
      const float costGiou);

  std::vector<std::pair<Tensor, Tensor>> compute(
      const Tensor& predBoxes,
      const Tensor& predLogits,
      const std::vector<Tensor>& targetBoxes,
      const std::vector<Tensor>& targetClasses) const;

 private:
  float costClass_;
  float costBbox_;
  float costGiou_;

  // First is SrcIdx, second is ColIdx
  std::pair<Tensor, Tensor> matchBatch(
      const Tensor& predBoxes,
      const Tensor& predLogits,
      const Tensor& targetBoxes,
      const Tensor& targetClasses) const;

  Tensor getCostMatrix(const Tensor& input, const Tensor& target);
};

} // namespace vision
} // namespace pkg
} // namespace fl
