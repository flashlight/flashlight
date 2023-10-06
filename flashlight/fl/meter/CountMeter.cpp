/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <stdexcept>

#include "flashlight/fl/meter/CountMeter.h"

namespace fl {

CountMeter::CountMeter(int num) : counts_(num, 0) {}

void CountMeter::add(int id, int64_t val) {
  if (!(id >= 0 && id < counts_.size())) {
    throw std::out_of_range("invalid id to update count for");
  }
  counts_[id] += val;
}

std::vector<int64_t> CountMeter::value() const {
  return counts_;
}

void CountMeter::reset() {
  std::fill(counts_.begin(), counts_.end(), 0);
}

} // namespace fl
