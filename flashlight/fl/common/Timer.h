/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>

namespace fl {

class Timer {
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;

 public:
  static Timer start();

  template <typename T = double>
  static T stop(const Timer& t) {
    return std::chrono::duration_cast<std::chrono::duration<T>>(
               std::chrono::high_resolution_clock::now() - t.startTime_)
        .count();
  }
};

} // namespace fl
