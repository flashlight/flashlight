/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/FullConnectionCriterion.h"

#include <cmath>

#include "flashlight/pkg/speech/criterion/CriterionUtils.h"

namespace fl::pkg::speech {

FullConnectionCriterion::FullConnectionCriterion(
    int N,
    fl::lib::seq::CriterionScaleMode scalemode)
    : N_(N), scaleMode_(scalemode) {
  if (N_ <= 0) {
    throw std::invalid_argument(
        "FCC: Size of transition matrix is less than 0.");
  }
  auto transition = constant(0.0, {N_, N_});
  params_ = {transition};
}

std::unique_ptr<Module> FullConnectionCriterion::clone() const {
  throw std::runtime_error(
      "Cloning is unimplemented in Module 'FullConnectionCriterion'");
}

std::string FullConnectionCriterion::prettyString() const {
  return "FullConnectionCriterion";
}
} // namespace fl
