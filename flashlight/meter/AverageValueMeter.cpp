/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/meter/AverageValueMeter.h"

#include <array>

namespace fl {

AverageValueMeter::AverageValueMeter() {
  reset();
}

void AverageValueMeter::reset() {
  curN_ = 0;
  curSum_ = 0.0;
  curVar_ = 0.0;
}

void AverageValueMeter::add(const double val, int64_t n /*= 1*/) {
  curSum_ += n * val;
  curVar_ += n * val * val;
  curN_ += n;
}

std::vector<double> AverageValueMeter::value() {
  double mean = (curN_ > 0) ? curSum_ / curN_ : 0;
  double var = (curN_ > 1) ? (curVar_ - curN_ * mean * mean) / (curN_ - 1) : 0;
  return {mean, var, static_cast<double>(curN_)};
}
} // namespace fl
