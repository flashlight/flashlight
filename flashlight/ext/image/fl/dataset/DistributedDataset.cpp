/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/image/fl/dataset/DistributedDataset.h"

#include <iostream>

namespace fl {
namespace ext {
namespace image {

DistributedDataset::DistributedDataset(
    std::shared_ptr<Dataset> base,
    int64_t worldRank,
    int64_t worldSize,
    int64_t batchSize,
    int64_t numThreads,
    int64_t prefetchSize,
    BatchDatasetPolicy batchpolicy,
    int64_t seed)
    : prefetchSize_(prefetchSize),
      numThreads_(numThreads),
      batchSize_(batchSize),
      batchpolicy_(batchpolicy) {
  base_ = std::make_shared<ShuffleDataset>(base, seed);
  auto permfn = [worldSize, worldRank](int64_t idx) {
    return (idx * worldSize) + worldRank;
  };

  int partitionSize = base_->size() / worldSize;
  int leftOver = base_->size() % worldSize;
  if (worldRank < leftOver) {
    partitionSize++;
  }
  base_ = std::make_shared<ResampleDataset>(base_, permfn, partitionSize);
  resample(seed);
}

std::vector<af::array> DistributedDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  return ds_->get(idx);
}

void DistributedDataset::resample(int64_t seed) {
  base_ = std::make_shared<ShuffleDataset>(base_, seed);
  ds_ = std::make_shared<BatchDataset>(base_, batchSize_, batchpolicy_);
  ds_ = std::make_shared<ShuffleDataset>(ds_, seed);
  ds_ = std::make_shared<PrefetchDataset>(ds_, numThreads_, prefetchSize_);
}

int64_t DistributedDataset::size() const {
  return ds_->size();
}

} // namespace image
} // namespace ext
} // namespace fl
