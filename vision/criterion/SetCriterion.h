#pragma once

#include <unordered_map>
#include <utility>

#include "Hungarian.h"

namespace fl {
namespace cv {

class SetCriterion {

public:
  using LossDict = std::unordered_map<std::string, Variable>;

  SetCriterion(
      const int num_classes, 
      const HungarianMatcher& matcher, 
      const af::array& weight_dict,
      const float eos_coef, 
      LossDict losses);

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
      const int numBoxes
      );

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
private:
  std::pair<af::array, af::array> getSrcPermutationIdx(
      const std::vector<std::pair<af::array, af::array>>& indices);

  std::pair<af::array, af::array> getTgtPermutationIdx(
      const std::vector<std::pair<af::array, af::array>>& indices);

  const int num_classes_;
  const HungarianMatcher matcher_;
  const af::array weight_dict_;
  const float eos_coef_;
  LossDict losses_;

};

} // end namespace cv
} // end namespace fl
