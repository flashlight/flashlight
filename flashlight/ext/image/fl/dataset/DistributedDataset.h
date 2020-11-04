/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/dataset/datasets.h"

namespace fl {
namespace ext {
namespace image {

class DistributedDataset : public Dataset {
 public:
  DistributedDataset(
      std::shared_ptr<Dataset> base,
      int64_t worldRank,
      int64_t worldSize,
      int64_t batchSize,
      int64_t numThreads,
      int64_t prefetchSize);

  std::vector<af::array> get(const int64_t idx) const override;

  void resample();

  int64_t size() const override;

 private:
  std::shared_ptr<Dataset> ds_;
  std::shared_ptr<ShuffleDataset> shuffle_;
};

} // namespace image
} // namespace ext
} // namespace fl
