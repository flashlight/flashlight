/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/speech/criterion/CriterionUtils.h"
#include "flashlight/pkg/speech/criterion/Defines.h"
#include "flashlight/pkg/speech/criterion/SequenceCriterion.h"

namespace fl {
namespace pkg {
namespace speech {

class ConnectionistTemporalClassificationCriterion : public SequenceCriterion {
 public:
  ConnectionistTemporalClassificationCriterion(
      fl::lib::seq::CriterionScaleMode scalemode =
          fl::lib::seq::CriterionScaleMode::NONE);

  std::unique_ptr<Module> clone() const override;

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  Tensor viterbiPath(const Tensor& input, const Tensor& inputSize = Tensor())
      override;

  Tensor viterbiPathWithTarget(
      const Tensor& input,
      const Tensor& target,
      const Tensor& inputSizes = Tensor(),
      const Tensor& targetSizes = Tensor()) override;

  std::string prettyString() const override;

 private:
  fl::lib::seq::CriterionScaleMode scaleMode_;

  FL_SAVE_LOAD_WITH_BASE(SequenceCriterion, scaleMode_)

  void validate(const fl::Variable& input, const fl::Variable& target);
};

typedef ConnectionistTemporalClassificationCriterion CTCLoss;
} // namespace speech
} // namespace pkg
} // namespace fl

CEREAL_REGISTER_TYPE(
    fl::pkg::speech::ConnectionistTemporalClassificationCriterion)
