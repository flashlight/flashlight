/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/speech/criterion/CriterionUtils.h"
#include "flashlight/pkg/speech/criterion/Defines.h"
#include "flashlight/pkg/speech/criterion/ForceAlignmentCriterion.h"
#include "flashlight/pkg/speech/criterion/FullConnectionCriterion.h"
#include "flashlight/pkg/speech/criterion/SequenceCriterion.h"

using fl::lib::seq::CriterionScaleMode;
namespace fl {
namespace app {
namespace asr {

class AutoSegmentationCriterion : public SequenceCriterion {
 public:
  explicit AutoSegmentationCriterion(
      int N,
      CriterionScaleMode scalemode = CriterionScaleMode::NONE,
      double transdiag = 0.0)
      : N_(N),
        scaleMode_(scalemode),
        fac_(ForceAlignmentCriterion(N, scalemode)),
        fcc_(FullConnectionCriterion(N, scalemode)) {
    if (N_ <= 0) {
      throw af::exception("ASG: N is zero or negative.");
    }
    fl::Variable transition(transdiag * af::identity(af::dim4(N_, N_)), true);
    params_ = {transition};
    syncTransitions();
  }

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override {
    if (inputs.size() != 2) {
      throw std::invalid_argument("Invalid inputs size");
    }
    return {fcc_.forward(inputs[0], inputs[1]) -
            fac_.forward(inputs[0], inputs[1])};
  }

  af::array viterbiPath(
      const af::array& input,
      const af::array& inputSize = af::array()) override {
    return fl::app::asr::viterbiPath(input, params_[0].array());
  }

  af::array viterbiPathWithTarget(
      const af::array& input,
      const af::array& target,
      const af::array& inputSizes = af::array(),
      const af::array& targetSizes = af::array()) override {
    return fac_.viterbiPath(input, target);
  }

  void setParams(const fl::Variable& var, int position) override {
    Module::setParams(var, position);
    syncTransitions();
  }

  std::string prettyString() const override {
    return "AutoSegmentationCriterion";
  }

 protected:
  AutoSegmentationCriterion() = default;

  void syncTransitions() {
    fac_.setParams(params_[0], 0);
    fcc_.setParams(params_[0], 0);
  }

 private:
  int N_;
  CriterionScaleMode scaleMode_;
  ForceAlignmentCriterion fac_;
  FullConnectionCriterion fcc_;

  FL_SAVE_LOAD_WITH_BASE(
      SequenceCriterion,
      fl::serializeAs<int64_t>(N_),
      scaleMode_,
      fac_,
      fcc_)
};

using ASGLoss = AutoSegmentationCriterion;
} // namespace asr
} // namespace app
} // namespace fl

CEREAL_REGISTER_TYPE(fl::app::asr::AutoSegmentationCriterion)
