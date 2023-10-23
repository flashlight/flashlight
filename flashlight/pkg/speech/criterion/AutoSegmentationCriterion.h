/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
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
namespace pkg {
namespace speech {

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
      throw std::invalid_argument("ASG: N is zero or negative.");
    }
    fl::Variable transition(transdiag * fl::identity(N_), true);
    params_ = {transition};
    syncTransitions();
  }

  std::unique_ptr<Module> clone() const override {
    throw std::runtime_error(
        "Cloning is unimplemented in Module 'AutoSegmentationCriterion'");
  }

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override {
    if (inputs.size() != 2) {
      throw std::invalid_argument("Invalid inputs size");
    }
    return {
        fcc_.forward(inputs[0], inputs[1]) -
        fac_.forward(inputs[0], inputs[1])};
  }

  Tensor viterbiPath(const Tensor& input, const Tensor& inputSize = Tensor())
      override {
    return fl::pkg::speech::viterbiPath(input, params_[0].tensor());
  }

  Tensor viterbiPathWithTarget(
      const Tensor& input,
      const Tensor& target,
      const Tensor& inputSizes = Tensor(),
      const Tensor& targetSizes = Tensor()) override {
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
} // namespace speech
} // namespace pkg
} // namespace fl

CEREAL_REGISTER_TYPE(fl::pkg::speech::AutoSegmentationCriterion)
