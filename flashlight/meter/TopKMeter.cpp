#include <iostream>
#include "flashlight/meter/TopKMeter.h"

namespace fl {

TopKMeter::TopKMeter(const int k, const bool accuracy)
    : k_(k), sum_(0), n_(0), accuracy_(accuracy){};

void TopKMeter::add(const af::array& output, const af::array& target) {
  if (output.dims(1) != target.dims(0)) {
    throw std::invalid_argument("dimension mismatch in TopKMeter");
  }
  if (target.numdims() != 1) {
    throw std::invalid_argument(
        "output/target must be 1-dimensional for TopKMeter");
  }

  af::array max_vals, max_ids, match;
  topk(max_vals, max_ids, output, k_, 0);
  match = batchFunc(
      max_ids, af::moddims(target, {1, target.dims(0), 1, 1}), af::operator==);
  const af::array correct = af::anyTrue(match, 0);

  uint64_t count = af::count<uint64_t>(correct);
  const int batchsize = target.dims(0);
  const double accuracy = static_cast<double>(count) * 100.f;
  sum_ += accuracy;
  n_ += batchsize;
}

void TopKMeter::reset() {
  sum_ = 0.0f;
  n_ = 0;
}

double TopKMeter::value() {
  double accuracy = (n_ > 0) ? (static_cast<double>(sum_) / n_) : 0.0;
  double val = (!accuracy_ ? (100.0 - accuracy) : accuracy);
  return val;
}

} // namespace fl
