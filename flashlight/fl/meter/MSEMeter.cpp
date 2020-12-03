/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/meter/MSEMeter.h"

namespace fl {
MSEMeter::MSEMeter() {
  reset();
}

void MSEMeter::reset() {
  curN_ = 0;
  curValue_ = .0;
}

void MSEMeter::add(const af::array& output, const af::array& target) {
  if (output.dims() != target.dims()) {
    throw std::invalid_argument("dimension mismatch in MSEMeter");
  }
  ++curN_;
  curValue_ = (curValue_ * (curN_ - 1) +
               af::sum<double>((output - target) * (output - target))) /
      curN_;
}

double MSEMeter::value() const {
  return curValue_;
}
} // namespace fl
