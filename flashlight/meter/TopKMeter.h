#include <arrayfire.h>

#pragma once

namespace fl {

/** TopKMeter computes the accuracy of the model outputs predicting the target
 * label in the top k predictions.
 *
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

class TopKMeter {
 public:
  /** Constructor of `TopKMeter`. Flag `accuracy` indicates if the meter
   * computes and returns accuracy or error rate instead. An instance will
   * maintain two counters initialized to 0:
   * - `n`: total samples
   * - `sum`: total mismatches
   */
  TopKMeter(const int k, const bool accuracy);

  void add(const af::array& output, const af::array& target);

  void reset();

  double value();

 private:
  int k_;
  int64_t sum_;
  int64_t n_;
  bool accuracy_;
};

} // namespace fl
