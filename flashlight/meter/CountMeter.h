/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>

namespace fl {
/** An implementation of count meter, which measures the total value of each
 * category.
 * Example usage:
 *
 * \code
 * CountMeter meter(10);  // 10 categories in total
 * for (intl id : data) {
 *   meter.add(id, 1);
 * }
 * std::vector<intl> count = meter.value();  // size of count should be 10
 * \endcode
 */
class CountMeter {
 public:
  /** Constructor of `CountMeter`. `num` specifies the total number of
   * catogories.
   */
  explicit CountMeter(intl num);

  /** Adds value `val` to category `id`. Note that `id` should be in range [0,
   * `num` - 1].*/
  void add(intl id, intl val);

  /** Returns a vector of `num` values, representing the total value of each
   * category.
   */
  std::vector<intl> value();

  /** Sets the value of each category to 0. */
  void reset();

 private:
  intl numCount_;
  std::vector<intl> countVal_;
};
} // namespace fl
