/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/meter/EditDistanceMeter.h"

#include <cstdlib> // TODO: whatever is actually needed for free?
#include <stdexcept>

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {
EditDistanceMeter::EditDistanceMeter() {
  reset();
}

void EditDistanceMeter::reset() {
  n_ = 0;
  ndel_ = 0;
  nins_ = 0;
  nsub_ = 0;
}

void EditDistanceMeter::add(const Tensor& output, const Tensor& target) {
  if (target.ndim() != 1) {
    throw std::invalid_argument(
        "target must be 1-dimensional for EditDistanceMeter");
  }
  if (output.ndim() != 1) {
    throw std::invalid_argument(
        "output must be 1-dimensional for EditDistanceMeter");
  }
  int len1 = output.dim(0);
  int len2 = target.dim(0);

  int* in1raw = output.host<int>();
  int* in2raw = target.host<int>();
  auto err_state = levensteinDistance(in1raw, in2raw, len1, len2);
  free(in1raw);
  in1raw = nullptr;
  free(in2raw);
  in2raw = nullptr;
  add(err_state, target.dim(0));
}

void EditDistanceMeter::add(
    const int64_t n,
    const int64_t ndel,
    const int64_t nins,
    const int64_t nsub) {
  n_ += n;
  ndel_ += ndel;
  nins_ += nins;
  nsub_ += nsub;
}

std::vector<int64_t> EditDistanceMeter::value() const {
  return {sumErr(), n_, ndel_, nins_, nsub_};
}

std::vector<double> EditDistanceMeter::errorRate() const {
  double val, valDel, valIns, valSub;
  if (n_ > 0) {
    val = static_cast<double>(sumErr() * 100.0) / n_;
    valDel = static_cast<double>(ndel_ * 100.0) / n_;
    valIns = static_cast<double>(nins_ * 100.0) / n_;
    valSub = static_cast<double>(nsub_ * 100.0) / n_;
  } else {
    val = (sumErr() > 0) ? std::numeric_limits<double>::infinity() : 0.0;
    valDel = (ndel_ > 0) ? std::numeric_limits<double>::infinity() : 0.0;
    valIns = (nins_ > 0) ? std::numeric_limits<double>::infinity() : 0.0;
    valSub = (nsub_ > 0) ? std::numeric_limits<double>::infinity() : 0.0;
  }
  return {val, static_cast<double>(n_), valDel, valIns, valSub};
}

} // namespace fl
