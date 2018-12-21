/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "EditDistanceMeter.h"
#include <flashlight/common/Exception.h>

#include <array>

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

void EditDistanceMeter::add(const af::array& output, const af::array& target) {
  AFML_ASSERT(
      target.numdims() == 1, "target must be 1 dimensional", target.numdims());
  AFML_ASSERT(
      output.numdims() == 1, "output must be 1 dimensional", output.numdims());
  int len1 = output.dims(0);
  int len2 = target.dims(0);

  int* in1raw = output.host<int>();
  int* in2raw = target.host<int>();
  auto err_state = levensteinDistance(in1raw, in2raw, len1, len2);
  af::freeHost(in1raw);
  af::freeHost(in2raw);
  add(err_state, target.dims(0));
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

std::vector<double> EditDistanceMeter::value() {
  double val = (n_ > 0) ? (static_cast<double>(sumErr() * 100.0) / n_) : 0.0;
  double val_del = (n_ > 0) ? (static_cast<double>(ndel_ * 100.0) / n_) : 0.0;
  double val_ins = (n_ > 0) ? (static_cast<double>(nins_ * 100.0) / n_) : 0.0;
  double val_sub = (n_ > 0) ? (static_cast<double>(nsub_ * 100.0) / n_) : 0.0;

  return {val, static_cast<double>(n_), val_del, val_ins, val_sub};
}
} // namespace fl
