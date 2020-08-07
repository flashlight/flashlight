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
#include "ForceAlignmentCriterion.h"
#include "FullConnectionCriterion.h"
#include "SequenceCriterion.h"

namespace fl {
namespace tasks {
namespace asr {

class AutoSegmentationCriterion : public SequenceCriterion {
 public:
  explicit AutoSegmentationCriterion(
      int N,
      lib::CriterionScaleMode scalemode = lib::CriterionScaleMode::NONE,
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

  af::array viterbiPath(const af::array& input) override {
    return fl::tasks::asr::viterbiPath(input, params_[0].array());
  }

  af::array viterbiPath(const af::array& input, const af::array& target)
      override {
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
  lib::CriterionScaleMode scaleMode_;
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
} // namespace tasks
} // namespace fl

CEREAL_REGISTER_TYPE(fl::tasks::asr::AutoSegmentationCriterion)
