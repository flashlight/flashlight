/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "flashlight/fl/common/Defines.h"

namespace fl {

class Tensor;

/**
 * @return if fp16 operations are supported with the current flashlight
 * configuration.
 */
FL_API bool f16Supported();

// Returns round-up result of integer division.
// throws invalid_argument exception on zero denominator.
FL_API size_t divRoundUp(size_t numerator, size_t denominator);

// Return a string formmated similar to: 1314127872(1GB+229MB+256KB)
FL_API std::string prettyStringMemorySize(size_t size);

// Returns a string formatted similar to: 26675644(2m+667k+5644)
FL_API std::string prettyStringCount(size_t count);

/**
 * Calls `f(args...)` repeatedly, retrying if an exception is thrown.
 * Supports sleeps between retries, with duration starting at `initial` and
 * multiplying by `factor` each retry. At most `maxIters` calls are made.
 */
template <class Fn, class... Args>
typename std::invoke_result<Fn, Args...>::type retryWithBackoff(
    std::chrono::duration<double> initial,
    double factor,
    int64_t maxIters,
    Fn&& f,
    Args&&... args) {
  if (!(initial.count() >= 0.0)) {
    throw std::invalid_argument("retryWithBackoff: bad initial");
  } else if (!(factor >= 0.0)) {
    throw std::invalid_argument("retryWithBackoff: bad factor");
  } else if (maxIters <= 0) {
    throw std::invalid_argument("retryWithBackoff: bad maxIters");
  }
  auto sleepSecs = initial.count();
  for (int64_t i = 0; i < maxIters; ++i) {
    try {
      return f(std::forward<Args>(args)...);
    } catch (...) {
      if (i >= maxIters - 1) {
        throw;
      }
    }
    if (sleepSecs > 0.0) {
      /* sleep override */
      std::this_thread::sleep_for(
          std::chrono::duration<double>(std::min(1e7, sleepSecs)));
    }
    sleepSecs *= factor;
  }
  throw std::logic_error("retryWithBackoff: hit unreachable");
}

/**
 * Get the value of an environment variable with a default value if not found.
 */
FL_API std::string getEnvVar(
    const std::string& key,
    const std::string& dflt = "");

/** @} */

} // namespace fl
