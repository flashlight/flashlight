/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/meter/MSEMeter.h"

#include <stdexcept>

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {
MSEMeter::MSEMeter() {
  reset();
}

void MSEMeter::reset() {
  curN_ = 0;
  curValue_ = .0;
}

void MSEMeter::add(const Tensor& output, const Tensor& target) {
  if (output.ndim() != target.ndim()) {
    throw std::invalid_argument("dimension mismatch in MSEMeter");
  }
  ++curN_;
  curValue_ =
      (curValue_ * (curN_ - 1) +
       fl::sum((output - target) * (output - target)).asScalar<double>()) /
      curN_;
}

double MSEMeter::value() const {
  return curValue_;
}
} // namespace fl
