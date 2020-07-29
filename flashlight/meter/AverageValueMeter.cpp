/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/meter/AverageValueMeter.h"

#include <array>

namespace fl {

AverageValueMeter::AverageValueMeter() {
  reset();
}

void AverageValueMeter::reset() {
  curMean_ = 0;
  curMeanSquaredSum_ = 0;
  curWeightSum_ = 0;
  curWeightSquaredSum_ = 0;
}

void AverageValueMeter::add(const double val, const double w /* = 1.0 */) {
  curWeightSum_ += w;
  curWeightSquaredSum_ += w * w;

  if (curWeightSum_ == 0) {
    return;
  }

  curMean_ = curMean_ + w * (val - curMean_) / curWeightSum_;
  curMeanSquaredSum_ =
      curMeanSquaredSum_ + w * (val * val - curMeanSquaredSum_) / curWeightSum_;
}

void AverageValueMeter::add(const af::array& vals) {
  double w = vals.elements();
  curWeightSum_ += w;
  curWeightSquaredSum_ += w;

  if (curWeightSum_ == 0) {
    return;
  }

  curMean_ = curMean_ + (af::sum<double>(vals) - w * curMean_) / curWeightSum_;
  curMeanSquaredSum_ = curMeanSquaredSum_ +
      (af::sum<double>(vals * vals) - w * curMeanSquaredSum_) / curWeightSum_;
}

std::vector<double> AverageValueMeter::value() {
  double mean = curMean_;
  double var = (curMeanSquaredSum_ - curMean_ * curMean_) /
      (1 - curWeightSquaredSum_ / (curWeightSum_ * curWeightSum_));
  return {mean, var, curWeightSum_};
}
} // namespace fl
