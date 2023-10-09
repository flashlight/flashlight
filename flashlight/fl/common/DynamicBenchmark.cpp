/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/DynamicBenchmark.h"
#include "flashlight/fl/tensor/Compute.h"

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
  fl::sync();
  currentTimer_ = fl::Timer::start();
}

void DynamicBenchmark::stop(bool incrementCount) {
  fl::sync();
  auto elapsedTime = fl::Timer::stop(currentTimer_);
  options_->accumulateTimeToCurrentOption(elapsedTime, incrementCount);
}

void DynamicBenchmark::setBenchmarkMode(bool mode) {
  benchmarkMode_ = mode;
}

bool DynamicBenchmark::getBenchmarkMode() {
  return benchmarkMode_;
}

} // namespace fl
