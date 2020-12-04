/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>

namespace fl {

/** An implementation of mean square error meter, which measures the mean square
 * error between targets and predictions made by the model.
 * Example usage:
 *
 * \code
 * MSEMeter meter();
 * for (auto& sample : data) {
 *   auto prediction = model(sample.input);
 *   meter.add(sample.target, prediction);
 * }
 * double mse = meter.value();
 * \endcode
 */
class MSEMeter {
 public:
  /** Constructor of `MSEMeter`. An instance will maintain two
   * counters initialized to 0:
   * - `n`: total samples
   * - `mse`: mean square error of samples
   */
  MSEMeter();

  /** Computes mean square error between two arrayfire arrays `output` and
   * `target` and updates the counters. Note that the shape of the two input
   * arrays should be identical.
   */
  void add(const af::array& output, const af::array& target);

  /** Returns a single value of mean square error. */
  double value() const;

  /** Sets all the counters to 0. */
  void reset();

 private:
  double curValue_;
  int64_t curN_;
};
} // namespace fl
