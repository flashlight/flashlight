/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/CriterionUtils.h"

#include <algorithm>
#include <cmath>

using fl::lib::seq::CriterionScaleMode;

namespace fl {
namespace app {
namespace asr {

int countRepeats(const int* labels, int len) {
  int r = 0;
  for (int i = 1; i < len; ++i) {
    if (labels[i] == labels[i - 1]) {
      ++r;
    }
  }
  return r;
}

int getTargetSize(const int* labels, int len) {
  while (len > 0 && labels[len - 1] < 0) {
    --len;
  }
  return len;
}

CriterionScaleMode getCriterionScaleMode(
    const std::string& onorm,
    bool sqnorm) {
  if (onorm == "none") {
    return CriterionScaleMode::NONE;
  } else if (onorm == "input") {
    return sqnorm ? CriterionScaleMode::INPUT_SZ_SQRT
                  : CriterionScaleMode::INPUT_SZ;
  } else if (onorm == "target") {
    return sqnorm ? CriterionScaleMode::TARGET_SZ_SQRT
                  : CriterionScaleMode::TARGET_SZ;
  } else {
    throw std::invalid_argument("invalid onorm option");
  }
}

Variable getLinearTarget(const Variable& targetVar, int T) {
  int L = targetVar.dims(0);
  int B = targetVar.dims(1);

  std::vector<int> target(B * L);
  std::vector<int> newTarget(B * T);

  targetVar.host(target.data());
  for (int b = 0; b < B; ++b) {
    const auto pTarget = target.data() + b * L;
    auto pNewTarget = newTarget.data() + b * T;

    int targetSize = std::min(T, fl::app::asr::getTargetSize(pTarget, L));
    if (targetSize == 0) {
      // hacky way to make ASG think L == 0.
      std::fill(pNewTarget, pNewTarget + T, -1);
    } else {
      for (int t = 0; t < T; ++t) {
        pNewTarget[t] = pTarget[t * targetSize / T];
      }
    }
  }
  return Variable(af::array(T, B, newTarget.data()), false);
}

fl::Variable applySeq2SeqMask(
    const fl::Variable& input,
    const af::array& targetClasses,
    int padValue) {
  if (input.dims() != targetClasses.dims()) {
    throw std::runtime_error(
        "applySeq2SeqMask: input and mask should have the same dimentions.");
  }
  af::array output = input.array();
  af::array mask = targetClasses == padValue;
  output(mask) = 0.;

  auto gradFunc = [mask](
                      std::vector<fl::Variable>& inputs,
                      const fl::Variable& gradOutput) {
    af::array gradArray = gradOutput.array();
    gradArray(mask) = 0.;
    inputs[0].addGrad(fl::Variable(gradArray, false));
  };
  return fl::Variable(output, {input.withoutData()}, gradFunc);
}

} // namespace asr
} // namespace app
} // namespace fl
