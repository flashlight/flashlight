/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/FullConnectionCriterion.h"

#include <cmath>

#include "flashlight/pkg/speech/criterion/CriterionUtils.h"

namespace fl {
namespace app {
namespace asr {

FullConnectionCriterion::FullConnectionCriterion(
    int N,
    fl::lib::seq::CriterionScaleMode scalemode)
    : N_(N), scaleMode_(scalemode) {
  if (N_ <= 0) {
    throw std::invalid_argument(
        "FCC: Size of transition matrix is less than 0.");
  }
  auto transition = constant(0.0, af::dim4(N_, N_));
  params_ = {transition};
}

std::string FullConnectionCriterion::prettyString() const {
  return "FullConnectionCriterion";
}
} // namespace asr
} // namespace app
} // namespace fl
