/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/speech/criterion/CriterionUtils.h"
#include "flashlight/pkg/speech/criterion/Defines.h"
#include "flashlight/pkg/speech/criterion/SequenceCriterion.h"

namespace fl {
namespace app {
namespace asr {

class ConnectionistTemporalClassificationCriterion : public SequenceCriterion {
 public:
  ConnectionistTemporalClassificationCriterion(
      fl::lib::seq::CriterionScaleMode scalemode =
          fl::lib::seq::CriterionScaleMode::NONE);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  af::array viterbiPath(
      const af::array& input,
      const af::array& inputSize = af::array()) override;

  af::array viterbiPathWithTarget(
      const af::array& input,
      const af::array& target,
      const af::array& inputSizes = af::array(),
      const af::array& targetSizes = af::array()) override;

  std::string prettyString() const override;

 private:
  fl::lib::seq::CriterionScaleMode scaleMode_;

  FL_SAVE_LOAD_WITH_BASE(SequenceCriterion, scaleMode_)

  void validate(const fl::Variable& input, const fl::Variable& target);
};

typedef ConnectionistTemporalClassificationCriterion CTCLoss;
} // namespace asr
} // namespace app
} // namespace fl

CEREAL_REGISTER_TYPE(fl::app::asr::ConnectionistTemporalClassificationCriterion)
