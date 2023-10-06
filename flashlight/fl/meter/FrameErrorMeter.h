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
 * An implementation of frame error meter, which measures the frame-level or
 * element-level mismatch between targets and predictions made by the model.
 * Example usage:
 *
 * \code
 * FrameErrorMeter meter();
 * for (auto& sample : data) {
 *   auto prediction = model(sample.input);
 *   meter.add(sample.target, prediction);
 * }
 * double frameErrorRate = meter.value();
 * \endcode
 */
class FL_API FrameErrorMeter {
 public:
  /** Constructor of `FrameErrorMeter`. Flag `accuracy` indicates if the meter
   * computes and returns accuracy or error rate instead. An instance will
   * maintain two counters initialized to 0:
   * - `n`: total samples
   * - `sum`: total mismatches
   */
  explicit FrameErrorMeter(bool accuracy = false);

  /** Computes frame-level mismatch between two arrayfire arrays `output` and
   * `target` and updates the counters. Note that the shape of the two input
   * arrays should be identical.
   */
  void add(const Tensor& output, const Tensor& target);

  /** Returns a single value in percentage. If `accuracy` is `True`, the value
   * returned is accuracy, error otherwise.
   */
  double value() const;

  /** Sets all the counters to 0. */
  void reset();

 private:
  std::int64_t n_;
  std::int64_t sum_;
  bool accuracy_;
};
} // namespace fl
