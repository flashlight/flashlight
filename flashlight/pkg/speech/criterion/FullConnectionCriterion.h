/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/flashlight.h"
#include "flashlight/pkg/speech/criterion/CriterionUtils.h"
#include "flashlight/pkg/speech/criterion/Defines.h"

namespace fl {
namespace pkg {
namespace speech {

class FullConnectionCriterion : public fl::BinaryModule {
 public:
  explicit FullConnectionCriterion(
      int N,
      fl::lib::seq::CriterionScaleMode scalemode =
          fl::lib::seq::CriterionScaleMode::NONE);

  std::unique_ptr<Module> clone() const override;

  fl::Variable forward(const fl::Variable& input, const fl::Variable& target)
      override;

  std::string prettyString() const override;

 private:
  friend class AutoSegmentationCriterion;
  FullConnectionCriterion() = default;

  int N_;
  fl::lib::seq::CriterionScaleMode scaleMode_;

  FL_SAVE_LOAD_WITH_BASE(
      fl::BinaryModule,
      fl::serializeAs<int64_t>(N_),
      scaleMode_)
};

typedef FullConnectionCriterion FCCLoss;
} // namespace speech
} // namespace pkg
} // namespace fl

CEREAL_REGISTER_TYPE(fl::pkg::speech::FullConnectionCriterion)
