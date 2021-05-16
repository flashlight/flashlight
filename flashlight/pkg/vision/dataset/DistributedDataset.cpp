/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/dataset/DistributedDataset.h"

namespace fl {
namespace ext {
namespace image {

DistributedDataset::DistributedDataset(
    std::shared_ptr<Dataset> base,
    int64_t worldRank,
    int64_t worldSize,
    int64_t batchSize,
    int64_t nRepeated,
    int64_t numThreads,
    int64_t prefetchSize,
    BatchDatasetPolicy batchPolicy) {
  shuffle_ = std::make_shared<ShuffleDataset>(base);
  auto permfn = [worldSize, worldRank, nRepeated](int64_t idx) {
    return (idx * worldSize + worldRank) / nRepeated;
  };

  int partitionSize = shuffle_->size() / worldSize;
  int leftOver = shuffle_->size() % worldSize;
  if (worldRank < leftOver) {
    partitionSize++;
  }
  ds_ = std::make_shared<ResampleDataset>(shuffle_, permfn, partitionSize);
  ds_ = std::make_shared<PrefetchDataset>(ds_, numThreads, prefetchSize);
  ds_ = std::make_shared<BatchDataset>(ds_, batchSize, batchPolicy);
}

std::vector<af::array> DistributedDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  return ds_->get(idx);
}

void DistributedDataset::resample(const int seed) {
  shuffle_->setSeed(seed);
  shuffle_->resample();
}

int64_t DistributedDataset::size() const {
  return ds_->size();
}

} // namespace image
} // namespace ext
} // namespace fl
