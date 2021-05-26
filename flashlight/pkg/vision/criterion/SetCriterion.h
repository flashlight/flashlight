/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
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

  std::vector<af::array> match(
      const Variable& predBoxes,
      const Variable& predLogits,
      const std::vector<Variable>& targetBoxes,
      const std::vector<Variable>& targetClasses);

  LossDict lossLabels(
      const Variable& predBoxes,
      const Variable& predLogits,
      const std::vector<Variable>& targetBoxes,
      const std::vector<Variable>& targetClasses,
      const std::vector<std::pair<af::array, af::array>>& indices,
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
      const std::vector<std::pair<af::array, af::array>>& indices,
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
  std::pair<af::array, af::array> getSrcPermutationIdx(
      const std::vector<std::pair<af::array, af::array>>& indices);

  std::pair<af::array, af::array> getTgtPermutationIdx(
      const std::vector<std::pair<af::array, af::array>>& indices);

  const int numClasses_;
  const HungarianMatcher matcher_;
  const std::unordered_map<std::string, float> weightDict_;
  const float eosCoef_;
};

} // namespace vision
} // namespace pkg
} // namespace fl
