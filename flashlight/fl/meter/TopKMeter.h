/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>
#include <utility>

#include "flashlight/fl/common/Defines.h"

namespace fl {

class Tensor;

/**
 * TopKMeter computes the accuracy of the model outputs predicting the target
 * label in the top k predictions.
 *
 * Example usage:
 *
 * \code
 * TopKMeter top5Meter(5);
 * for (auto& sample : data) {
 *   auto prediction = model(sample.input);
 *   meter.add(sample.target, prediction);
 * }
 * double top5Accuracy = top5Meter.value();
 * \endcode
 */
class FL_API TopKMeter {
 public:
  /** Constructor of `TopKMeter`.
   * @param k number of top predictions in order to be considered correct
   * Will have two counters:
   * - `correct`: total number of correct predictions
   * - `n`: total number of of predictions
   */
  explicit TopKMeter(const int k);

  void add(const Tensor& output, const Tensor& target);

  void reset();

  // Used for distributed syncing
  void set(int32_t correct, int32_t n);

  std::pair<int32_t, int32_t> getStats();

  double value() const;

 private:
  int k_;
  int32_t correct_;
  int32_t n_;
};

} // namespace fl
