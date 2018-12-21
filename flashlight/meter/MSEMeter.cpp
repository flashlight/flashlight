/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "MSEMeter.h"
#include <flashlight/common/Exception.h>

namespace fl {
MSEMeter::MSEMeter() {
  reset();
}

void MSEMeter::reset() {
  curN_ = 0;
  curValue_ = .0;
}

void MSEMeter::add(const af::array& output, const af::array& target) {
  AFML_ASSERT(
      output.dims() == target.dims(), "Dimension mismatch in MSEMeter", -1);
  ++curN_;
  curValue_ = (curValue_ * (curN_ - 1) +
               af::sum<double>((output - target) * (output - target))) /
      curN_;
}

double MSEMeter::value() {
  return curValue_;
}
} // namespace fl
