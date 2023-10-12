/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>

#include "flashlight/fl/common/Defines.h"

namespace fl {

/**
 * An implementation of timer, which measures the wall clock time.
 * Example usage:
 *
 * \code
 * TimeMeter meter();
 * meter.resume();
 * // Do something here;
 * meter.stop();
 * double time = meter.value();
 * \endcode
 */
class FL_API TimeMeter {
 public:
  /** Constructor of `TimeMeter`. An instance will maintain a timer which is
   * initialized as stopped. The flag `unit` indicates if there is multiple
   * units running in sequential in the current timing period.
   */
  explicit TimeMeter(bool unit = false);

  /** Stops the timer if still running. If `unit` is `True`, returns the average
   * time spend per unit, otherwise the total time in the current timing period.
   * Time is measured in seconds.
   */
  double value() const;

  /** Refreshes the counters and stops the timer. */
  void reset();

  /** Increases the number of units by `num`. */
  void incUnit(int64_t num = 1);

  /** Starts the timer. */
  void resume();

  /** Stops the timer. */
  void stop();

  /** Sets the number of units by `num` and the total time spend by `val`. */
  void set(double val, int64_t num = 1);

  /** Stops the timer and increase the number of units by `num`. */
  void stopAndIncUnit(int64_t num = 1);

 private:
  std::chrono::time_point<std::chrono::system_clock> start_;
  double curValue_;
  int64_t curN_;
  bool isStopped_;
  bool useUnit_;
};
} // namespace fl
