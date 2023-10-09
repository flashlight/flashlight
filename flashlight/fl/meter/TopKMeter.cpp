/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "flashlight/fl/meter/TopKMeter.h"

#include <stdexcept>

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

TopKMeter::TopKMeter(const int k) : k_(k), correct_(0), n_(0){};

void TopKMeter::add(const Tensor& output, const Tensor& target) {
  if (output.dim(1) != target.dim(0)) {
    throw std::invalid_argument("dimension mismatch in TopKMeter");
  }
  if (target.ndim() != 1) {
    throw std::invalid_argument(
        "output/target must be 1-dimensional for TopKMeter");
  }

  Tensor maxVals, maxIds, match;
  topk(maxVals, maxIds, output, k_, 0);
  match = maxIds == fl::reshape(target, {1, target.dim(0), 1, 1});
  const Tensor correct = fl::any(match, {0});

  correct_ += fl::countNonzero(correct).asScalar<int32_t>();
  const int batchsize = target.dim(0);
  n_ += batchsize;
}

void TopKMeter::reset() {
  correct_ = 0;
  n_ = 0;
}

double TopKMeter::value() const {
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
