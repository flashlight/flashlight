/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace fl {
/** An implementation of count meter, which measures the total value of each
 * category.
 * Example usage:
 *
 * \code
  CountMeter meter(10);  // 10 categories in total
  meter.add(4, 6);  // add 6 count to category 4
  meter.add(7, 2);  // add 2 count to category 7
  meter.add(4, -1);  // add -1 count to category 4

  auto counts = meter.value();
  std::cout << counts[4];  // prints 5
  \endcode
 */
class CountMeter {
 public:
  /** Constructor of `CountMeter`. `num` specifies the total number of
   * categories.
   */
  explicit CountMeter(int num);

  /** Adds value `val` to category `id`. Note that `id` should be in range [0,
   * `num` - 1].*/
  void add(int id, int64_t val);

  /** Returns a vector of `num` values, representing the total value of each
   * category.
   */
  std::vector<int64_t> value() const;

  /** Sets the value of each category to 0. */
  void reset();

 private:
  std::vector<int64_t> counts_;
};
} // namespace fl
