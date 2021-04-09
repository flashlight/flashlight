/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/meter/TimeMeter.h"

namespace fl {

TimeMeter::TimeMeter(bool unit /* = false */) : useUnit_(unit) {
  reset();
}

void TimeMeter::reset() {
  curN_ = 0;
  curValue_ = 0.;
  isStopped_ = true;
}

void TimeMeter::set(double val, int64_t num /* = 1 */) {
  curValue_ = val;
  curN_ = num;
  start_ = std::chrono::system_clock::now();
}

double TimeMeter::value() const {
  double val = curValue_;
  if (!isStopped_) {
    std::chrono::duration<double> duration =
        std::chrono::system_clock::now() - start_;
    val += duration.count();
  }
  if (useUnit_) {
    val = (curN_ > 0) ? (val / curN_) : 0.0;
  }
  return val;
}

void TimeMeter::stop() {
  if (isStopped_) {
    return;
  }
  std::chrono::duration<double> duration =
      std::chrono::system_clock::now() - start_;
  curValue_ += duration.count();
  isStopped_ = true;
}

void TimeMeter::resume() {
  if (!isStopped_) {
    return;
  }
  start_ = std::chrono::system_clock::now();
  isStopped_ = false;
}

void TimeMeter::incUnit(int64_t num) {
  curN_ += num;
}

void TimeMeter::stopAndIncUnit(int64_t num) {
  stop();
  incUnit(num);
}
} // namespace fl
