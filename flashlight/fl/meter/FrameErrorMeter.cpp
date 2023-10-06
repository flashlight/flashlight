/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/meter/FrameErrorMeter.h"

#include <stdexcept>

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {
FrameErrorMeter::FrameErrorMeter(bool accuracy /* = false */)
    : accuracy_(accuracy) {
  reset();
}

void FrameErrorMeter::reset() {
  n_ = 0;
  sum_ = 0;
}

void FrameErrorMeter::add(const Tensor& output, const Tensor& target) {
  if (output.shape() != target.shape()) {
    throw std::invalid_argument("dimension mismatch in FrameErrorMeter");
  }
  if (target.ndim() != 1) {
    throw std::invalid_argument(
        "output/target must be 1-dimensional for FrameErrorMeter");
  }

  sum_ += fl::countNonzero(output != target).scalar<unsigned>();
  n_ += target.dim(0);
}

double FrameErrorMeter::value() const {
  double error = (n_ > 0) ? (static_cast<double>(sum_ * 100.0) / n_) : 0.0;
  double val = (accuracy_ ? (100.0 - error) : error);
  return val;
}
} // namespace fl
