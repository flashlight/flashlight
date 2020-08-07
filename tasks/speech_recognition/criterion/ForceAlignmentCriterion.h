/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

#include "flashlight/tasks/speech_recognition/criterion/CriterionUtils.h"
#include "flashlight/tasks/speech_recognition/criterion/Defines.h"

namespace fl {
namespace tasks {
namespace asr {

class ForceAlignmentCriterion : public fl::BinaryModule {
 public:
  explicit ForceAlignmentCriterion(
      int N,
      lib::CriterionScaleMode scalemode = fl::lib::CriterionScaleMode::NONE);

  fl::Variable forward(const fl::Variable& input, const fl::Variable& target)
      override;

  af::array viterbiPath(const af::array& input, const af::array& target);

  std::string prettyString() const override;

 private:
  friend class AutoSegmentationCriterion;
  ForceAlignmentCriterion() = default;

  int N_;
  fl::lib::CriterionScaleMode scaleMode_;

  FL_SAVE_LOAD_WITH_BASE(
      fl::BinaryModule,
      fl::serializeAs<int64_t>(N_),
      scaleMode_)
};

typedef ForceAlignmentCriterion FACLoss;
} // namespace asr
} // namespace tasks
} // namespace fl

CEREAL_REGISTER_TYPE(fl::tasks::asr::ForceAlignmentCriterion)
