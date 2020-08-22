/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "AutoSegmentationCriterion.h"
#include "CriterionUtils.h"

using fl::lib::seq::CriterionScaleMode;

namespace fl {
namespace app {
namespace asr {

class LinearSegmentationCriterion : public AutoSegmentationCriterion {
 public:
  explicit LinearSegmentationCriterion(
      int N,
      CriterionScaleMode scaleMode = CriterionScaleMode::NONE)
      : AutoSegmentationCriterion(N, scaleMode) {}

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override {
    if (inputs.size() != 2) {
      throw std::invalid_argument("Invalid inputs size");
    }
    const auto& input = inputs[0];
    const auto& target = inputs[1];
    return AutoSegmentationCriterion::forward(
        {input, getLinearTarget(target, input.dims(1))});
  }

  std::string prettyString() const override {
    return "LinearSegmentationCriterion";
  }

 private:
  LinearSegmentationCriterion() = default;

  FL_SAVE_LOAD_WITH_BASE(AutoSegmentationCriterion)
};

using LinSegCriterion = LinearSegmentationCriterion;
} // namespace asr
} // namespace app
} // namespace fl

CEREAL_REGISTER_TYPE(fl::app::asr::LinearSegmentationCriterion)
