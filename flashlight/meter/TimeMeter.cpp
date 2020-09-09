/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/flashlight/meter/TimeMeter.h"

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
}

double TimeMeter::value() {
  if (!isStopped_) {
    curValue_ += af::timer::stop(curTimer_);
    curTimer_ = af::timer::start();
  }
  double val;
  if (useUnit_) {
    val = (curN_ > 0) ? (curValue_ / curN_) : 0.0;
  } else {
    val = curValue_;
  }
  return val;
}

void TimeMeter::stop() {
  if (isStopped_) {
    return;
  }
  curValue_ += af::timer::stop(curTimer_);
  isStopped_ = true;
}

void TimeMeter::resume() {
  if (!isStopped_) {
    return;
  }
  curTimer_ = af::timer::start();
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
