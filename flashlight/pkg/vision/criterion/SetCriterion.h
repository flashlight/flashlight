/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include <utility>

#include "flashlight/pkg/vision/criterion/Hungarian.h"
#include "flashlight/fl/autograd/Variable.h"

namespace fl {
namespace pkg {
namespace vision {

class SetCriterion {
 public:
  using LossDict = std::unordered_map<std::string, Variable>;

  SetCriterion(
      const int numClasses,
      const HungarianMatcher& matcher,
      const std::unordered_map<std::string, float>& weightDict,
      const float eosCoef);

  std::vector<Tensor> match(
      const Variable& predBoxes,
      const Variable& predLogits,
      const std::vector<Variable>& targetBoxes,
      const std::vector<Variable>& targetClasses);

  LossDict lossLabels(
      const Variable& predBoxes,
      const Variable& predLogits,
      const std::vector<Variable>& targetBoxes,
      const std::vector<Variable>& targetClasses,
      const std::vector<std::pair<Tensor, Tensor>>& indices,
      const int numBoxes);

  LossDict lossCardinality(
      const Variable& predBoxes,
      const Variable& predLogits,
      const std::vector<Variable>& targetBoxes,
      const std::vector<Variable>& targetClasses);

  LossDict lossBoxes(
      const Variable& predBoxes,
      const Variable& predLogits,
      const std::vector<Variable>& targetBoxes,
      const std::vector<Variable>& targetClasses,
      const std::vector<std::pair<Tensor, Tensor>>& indices,
      const int numBoxes);

  LossDict lossMasks(
      const Variable& predBoxes,
      const Variable& predLogits,
      const std::vector<Variable>& targetBoxes,
      const std::vector<Variable>& targetClasses);

  LossDict forward(
      const Variable& predBoxes,
      const Variable& predLogits,
      const std::vector<Variable>& targetBoxes,
      const std::vector<Variable>& targetClasses);

  std::unordered_map<std::string, float> getWeightDict();

 private:
  std::pair<Tensor, Tensor> getSrcPermutationIdx(
      const std::vector<std::pair<Tensor, Tensor>>& indices);

  std::pair<Tensor, Tensor> getTgtPermutationIdx(
      const std::vector<std::pair<Tensor, Tensor>>& indices);

  const int numClasses_;
  const HungarianMatcher matcher_;
  const std::unordered_map<std::string, float> weightDict_;
  const float eosCoef_;
};

} // namespace vision
} // namespace pkg
} // namespace fl
