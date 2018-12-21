/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>

namespace fl {
/** An implementation of average value meter, which measures the mean and
 * variance of a sequence of values.
 * Example usage:
 *
 * \code
 * AverageValueMeter meter();
 * for (double sample : data) {
 *   meter.add(sample);
 * }
 * double mean = meter.value()[0];
 * \endcode
 */
class AverageValueMeter {
 public:
  /** Constructor of `AverageValueMeter`. An instance will maintain three
   * counters initialized to 0:
   * - `n`: total number of values
   * - `sum`: sum of the values
   * - `squared_sum`: sum of the square of values
   */
  AverageValueMeter();

  /** Updates counters with the given value `val` and its repetition `n`. */
  void add(const double val, int64_t n = 1);

  /** Returns a vector of three values:
   * - `mean`: \f$ \frac{\sum_{i = 1}^n x_i}{n} \f$
   * - `variance`: \f$ \frac{\sum_{i = 1}^n (x_i - mean)^2}{n - 1} \f$
   * - `N`: \f$ n \f$
   */
  std::vector<double> value();

  /** Sets all the counters to 0. */
  void reset();

 private:
  double curSum_, curVar_;
  int64_t curN_;
};
} // namespace fl
