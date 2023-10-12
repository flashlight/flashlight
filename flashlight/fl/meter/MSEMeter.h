/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include "flashlight/fl/common/Defines.h"

namespace fl {

class Tensor;

/**
 * An implementation of mean square error meter, which measures the mean square
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
class FL_API MSEMeter {
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
  void add(const Tensor& output, const Tensor& target);

  /** Returns a single value of mean square error. */
  double value() const;

  /** Sets all the counters to 0. */
  void reset();

 private:
  double curValue_;
  int64_t curN_;
};
} // namespace fl
