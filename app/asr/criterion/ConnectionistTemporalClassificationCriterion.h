/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "CriterionUtils.h"
#include "Defines.h"
#include "SequenceCriterion.h"

using fl::lib::seq::CriterionScaleMode;

namespace fl {
namespace tasks {
namespace asr {

class ConnectionistTemporalClassificationCriterion : public SequenceCriterion {
 public:
  ConnectionistTemporalClassificationCriterion(
      CriterionScaleMode scalemode = CriterionScaleMode::NONE);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  af::array viterbiPath(const af::array& input) override;
  af::array viterbiPath(const af::array& input, const af::array& target)
      override;

  std::string prettyString() const override;

 private:
  CriterionScaleMode scaleMode_;

  FL_SAVE_LOAD_WITH_BASE(SequenceCriterion, scaleMode_)

  void validate(const fl::Variable& input, const fl::Variable& target);
};

typedef ConnectionistTemporalClassificationCriterion CTCLoss;
} // namespace asr
} // namespace tasks
} // namespace fl

CEREAL_REGISTER_TYPE(
    fl::tasks::asr::ConnectionistTemporalClassificationCriterion)
