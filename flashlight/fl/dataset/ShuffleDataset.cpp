/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/dataset/ShuffleDataset.h"

#include <algorithm>
#include <numeric>

namespace fl {

ShuffleDataset::ShuffleDataset(
    std::shared_ptr<const Dataset> dataset,
    int seed /* = 0 */)
    : ResampleDataset(dataset), rng_(seed) {
  resample();
}

void ShuffleDataset::resample() {
  std::iota(resampleVec_.begin(), resampleVec_.end(), 0);
  auto n = resampleVec_.size();
  // custom implementation of shuffle -
  // en.cppreference.com/w/cpp/algorithm/random_shuffle#Possible_implementation
  using distr_t = std::uniform_int_distribution<unsigned int>;
  distr_t D;
  for (int i = n - 1; i > 0; --i) {
    std::swap(
        resampleVec_[i], resampleVec_[D(rng_, distr_t::param_type(0, i))]);
  }
}

void ShuffleDataset::setSeed(int seed) {
  rng_.seed(seed);
}

} // namespace fl
