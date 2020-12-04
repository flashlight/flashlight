/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/meter/FrameErrorMeter.h"

namespace fl {
FrameErrorMeter::FrameErrorMeter(bool accuracy /* = false */)
    : accuracy_(accuracy) {
  reset();
}

void FrameErrorMeter::reset() {
  n_ = 0;
  sum_ = 0;
}

void FrameErrorMeter::add(const af::array& output, const af::array& target) {
  if (output.dims() != target.dims()) {
    throw std::invalid_argument("dimension mismatch in FrameErrorMeter");
  }
  if (target.numdims() != 1) {
    throw std::invalid_argument(
        "output/target must be 1-dimensional for FrameErrorMeter");
  }

  sum_ += af::count<int64_t>(output != target);
  n_ += target.dims(0);
}

double FrameErrorMeter::value() const {
  double error = (n_ > 0) ? (static_cast<double>(sum_ * 100.0) / n_) : 0.0;
  double val = (accuracy_ ? (100.0 - error) : error);
  return val;
}
} // namespace fl
