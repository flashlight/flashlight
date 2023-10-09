/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/ForceAlignmentCriterion.h"

namespace fl::pkg::speech {

ForceAlignmentCriterion::ForceAlignmentCriterion(
    int N,
    fl::lib::seq::CriterionScaleMode scalemode)
    : N_(N), scaleMode_(scalemode) {
  if (N_ <= 0) {
    throw std::invalid_argument(
        "FAC: Size of transition matrix is less than 0");
  }
  auto transition = fl::constant(0.0, {N_, N_});
  params_ = {transition};
}

std::unique_ptr<Module> ForceAlignmentCriterion::clone() const {
  throw std::runtime_error(
      "Cloning is unimplemented in Module 'ForceAlignmentCriterion'");
}

std::string ForceAlignmentCriterion::prettyString() const {
  return "ForceAlignmentCriterion";
}
} // namespace fl
