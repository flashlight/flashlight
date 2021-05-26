/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/ForceAlignmentCriterion.h"

namespace fl {
namespace pkg {
namespace speech {

ForceAlignmentCriterion::ForceAlignmentCriterion(
    int N,
    fl::lib::seq::CriterionScaleMode scalemode)
    : N_(N), scaleMode_(scalemode) {
  if (N_ <= 0) {
    throw std::invalid_argument(
        "FAC: Size of transition matrix is less than 0");
  }
  auto transition = fl::constant(0.0, af::dim4(N_, N_));
  params_ = {transition};
}

std::string ForceAlignmentCriterion::prettyString() const {
  return "ForceAlignmentCriterion";
}
} // namespace speech
} // namespace pkg
} // namespace fl
