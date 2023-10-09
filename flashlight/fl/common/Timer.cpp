/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/Timer.h"

namespace fl {

Timer Timer::start() {
  Timer t;
  t.startTime_ = std::chrono::high_resolution_clock::now();
  return t;
}

} // namespace fl
