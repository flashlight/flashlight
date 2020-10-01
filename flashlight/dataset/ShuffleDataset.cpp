/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/flashlight/dataset/ShuffleDataset.h"

#include <algorithm>

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
  // custom implementation of shuffle - https://stackoverflow.com/a/51931164
  for (auto i = n; i >= 1; --i) {
    std::swap(resampleVec_[i - 1], resampleVec_[rng_() % n]);
  }
}

void ShuffleDataset::setSeed(int seed) {
  rng_.seed(seed);
}

} // namespace fl
