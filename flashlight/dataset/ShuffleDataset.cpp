/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/dataset/ShuffleDataset.h"

#include <algorithm>

namespace fl {

ShuffleDataset::ShuffleDataset(std::shared_ptr<const Dataset> dataset)
    : ResampleDataset(dataset), rng_(std::default_random_engine(0)) {
  resample();
}

void ShuffleDataset::resample() {
  std::iota(resampleVec_.begin(), resampleVec_.end(), 0);
  std::shuffle(resampleVec_.begin(), resampleVec_.end(), rng_);
}

void ShuffleDataset::setSeed(int seed) {
  rng_.seed(seed);
}

} // namespace fl
