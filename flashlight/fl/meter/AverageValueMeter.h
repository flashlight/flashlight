/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>

namespace fl {
/** An implementation of average value meter, which measures the mean and
 * variance of a sequence of values.
 *
 * This meter takes as input a stream of i.i.d data \f$ X = {x_i} \f$ with
 * unormalized weights \f$ W = {w_i} \f$ (\f$ w_i \ge 0 \f$). Suppose \f$ p_i =
 * w_i / \sum_{j = 1}^n w_j \f$, it maintains the following variables:
 *
 * - unbiased mean \f$ \tilde{mu} = \sum_{i = 1}^n p_i x_i \f$
 * - unbiased second momentum \f$ \tilde{mu}_2 = \sum_{i = 1}^n p_i x_i^2 \f$
 * - sum of weights \f$ Sum(W) = \sum_{i = 1}^n w_i} \f$
 * - sum of squared weights \f$ Sum(W^2) = \sum_{i = 1}^n w_i^2} \f$
 *
 * Thus, we have \f$ Sum(P^2) = \sum_{i = 1}^n p_i^2} = Sum(W^2) / Sum(W)^2 \f$.
 *
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
  /** Constructor of `AverageValueMeter`. */
  AverageValueMeter();

  /** Updates counters with the given value `val` with weight `w`. */
  void add(const double val, const double w = 1.0);

  /** Updates counters with all values in `vals` with equal weights. */
  void add(const af::array& vals);

  /** Returns a vector of four values:
   * - `unbiased mean`: \f$ \tilde{mu} \f$
   * - `unbiased variance`: \f$ \tilde{sigma}^2 = \frac{(\tilde{mu}_2 -
   * \tilde{mu}^2)}{1 - Sum(P^2)} \f$
   * - `weight_sum`: \f$ Sum(W) \f$
   * - `weight_squared_sum`: \f$ Sum(W^2) \f$
   */
  std::vector<double> value() const;

  /** Sets all the counters to 0. */
  void reset();

 private:
  double curMean_;
  double curMeanSquaredSum_;
  double curWeightSum_;
  double curWeightSquaredSum_;
};
} // namespace fl
