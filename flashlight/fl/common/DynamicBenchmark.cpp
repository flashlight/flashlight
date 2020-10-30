/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/DynamicBenchmark.h"

namespace fl {

// Default value for benchmark mode
bool DynamicBenchmark::benchmarkMode_ = false;

void DynamicBenchmark::audit(
    const std::function<void()>& function,
    bool incrementCount) {
  // Only run the benchmarking components if some options are yet to be
  // fully-timed and benchmark mode is on - otherwise, only run the passed
  // lambda
  if (options_->timingsComplete() || !benchmarkMode_) {
    function();
  } else {
    start();
    function();
    stop(incrementCount);
  }
}

void DynamicBenchmark::start() {
  af::sync();
  currentTimer_ = af::timer::start();
}

void DynamicBenchmark::stop(bool incrementCount) {
  af::sync();
  auto elapsedTime = af::timer::stop(currentTimer_);
  options_->accumulateTimeToCurrentOption(elapsedTime, incrementCount);
}

void DynamicBenchmark::setBenchmarkMode(bool mode) {
  benchmarkMode_ = mode;
}

bool DynamicBenchmark::getBenchmarkMode() {
  return benchmarkMode_;
}

} // namespace fl
