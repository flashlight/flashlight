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
      int64_t prefetchSize,
      BatchDatasetPolicy batchpolicy = fl::BatchDatasetPolicy::INCLUDE_LAST,
      int64_t seed = 0);

  std::vector<af::array> get(const int64_t idx) const override;

  void resample(int64_t seed = 0);

  int64_t size() const override;

 private:
  int64_t prefetchSize_;
  int64_t numThreads_;
  int64_t batchSize_;
  BatchDatasetPolicy batchpolicy_;
  std::shared_ptr<Dataset> base_;
  std::shared_ptr<Dataset> ds_;
};

} // namespace image
} // namespace ext
} // namespace fl
