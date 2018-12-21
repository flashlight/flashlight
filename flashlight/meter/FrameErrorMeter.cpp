/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "FrameErrorMeter.h"
#include <flashlight/common/Exception.h>

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
  AFML_ASSERT(
      output.dims() == target.dims(),
      "target and output dimensions do not match",
      -1);
  AFML_ASSERT(
      target.numdims() == 1,
      "output, target must be 1 dimensional",
      target.numdims());

  sum_ += af::count<int64_t>(output != target);
  n_ += target.dims(0);
}

double FrameErrorMeter::value() {
  double error = (n_ > 0) ? (static_cast<double>(sum_ * 100.0) / n_) : 0.0;
  double val = (accuracy_ ? (100.0 - error) : error);
  return val;
}
} // namespace fl
