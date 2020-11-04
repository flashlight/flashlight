/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "flashlight/fl/meter/TopKMeter.h"

#include <iostream>
#include <stdexcept>

namespace fl {

TopKMeter::TopKMeter(const int k)
    : k_(k), correct_(0), n_(0) {};

void TopKMeter::add(const af::array& output, const af::array& target) {
  if (output.dims(1) != target.dims(0)) {
    throw std::invalid_argument("dimension mismatch in TopKMeter");
  }
  if (target.numdims() != 1) {
    throw std::invalid_argument(
        "output/target must be 1-dimensional for TopKMeter");
  }

  af::array maxVals, maxIds, match;
  topk(maxVals, maxIds, output, k_, 0);
  match = af::batchFunc(
      maxIds, af::moddims(target, {1, target.dims(0), 1, 1}), af::operator==);
  const af::array correct = af::anyTrue(match, 0);

  correct_ += af::count<int32_t>(correct);
  const int batchsize = target.dims(0);
  n_ += batchsize;
}

void TopKMeter::reset() {
  correct_ = 0;
  n_ = 0;
}

double TopKMeter::value() {
  return (static_cast<double>(correct_) / n_) * 100.0f;
}

std::pair<int32_t, int32_t> TopKMeter::getStats() {
  return std::make_pair(correct_, n_);
}

void TopKMeter::set(int32_t correct, int32_t n) {
  n_ = n;
  correct_ = correct;
}

} // namespace fl
