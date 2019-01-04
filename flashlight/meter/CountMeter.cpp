/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/meter/CountMeter.h"

namespace fl {

CountMeter::CountMeter(intl num) : numCount_(num) {
  reset();
}

void CountMeter::reset() {
  countVal_ = std::vector<intl>(numCount_, 0.0);
}

void CountMeter::add(intl id, intl val) {
  if (!(id >= 0 && id < numCount_)) {
    throw std::invalid_argument("invalid idx to update count for");
  }
  countVal_[id] += val;
}

std::vector<intl> CountMeter::value() {
  return countVal_;
}
} // namespace fl
