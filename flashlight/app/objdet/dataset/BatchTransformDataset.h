/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/dataset/datasets.h"

#include <cmath>

namespace fl {
namespace app {
namespace objdet {

template <class T>
using BatchTransformFunction =
    std::function<T(const std::vector<std::vector<af::array>>&)>;

/*
 * This is a slightly more generalized batching dataset than allows you to
 * return any type from the batched arrays. This is useful for Object detection
 * because we would like to keep the target boxes and classes as a separate
 * unbatched vector of arrays, while still batching the images
 */
template <typename T>
class BatchTransformDataset {
 public:
  BatchTransformDataset(
      std::shared_ptr<const Dataset> dataset,
      int64_t batchsize,
      BatchDatasetPolicy policy /* = BatchDatasetPolicy::INCLUDE_LAST */,
      BatchTransformFunction<T> batchFn)
      : dataset_(dataset),
        batchSize_(batchsize),
        batchPolicy_(policy),
        batchFn_(batchFn) {
    if (!dataset_) {
      throw std::invalid_argument("dataset to be batched is null");
    }
    if (batchSize_ <= 0) {
      throw std::invalid_argument("invalid batch size");
    }
    preBatchSize_ = dataset_->size();
    switch (batchPolicy_) {
      case BatchDatasetPolicy::INCLUDE_LAST:
        size_ = std::ceil(static_cast<double>(preBatchSize_) / batchSize_);
        break;
      case BatchDatasetPolicy::SKIP_LAST:
        size_ = std::floor(static_cast<double>(preBatchSize_) / batchSize_);
        break;
      case BatchDatasetPolicy::DIVISIBLE_ONLY:
        if (size_ % batchSize_ != 0) {
          throw std::invalid_argument(
              "dataset is not evenly divisible into batches");
        }
        size_ = std::ceil(static_cast<double>(preBatchSize_) / batchSize_);
        break;
      default:
        throw std::invalid_argument("unknown BatchDatasetPolicy");
    }
  }

  ~BatchTransformDataset() {}

  T get(const int64_t idx) {
    if (!(idx >= 0 && idx < size())) {
      throw std::out_of_range("Dataset idx out of range");
    }
    std::vector<std::vector<af::array>> buffer;

    int64_t start = batchSize_ * idx;
    int64_t end = std::min(start + batchSize_, preBatchSize_);

    for (int64_t batchidx = start; batchidx < end; ++batchidx) {
      auto fds = dataset_->get(batchidx);
      if (buffer.size() < fds.size()) {
        buffer.resize(fds.size());
      }
      for (int64_t i = 0; i < fds.size(); ++i) {
        buffer[i].emplace_back(fds[i]);
      }
    }
    return batchFn_(buffer);
  }

  int64_t size() const {
    return size_;
  }

 private:
  std::shared_ptr<const Dataset> dataset_;
  int64_t batchSize_;
  BatchDatasetPolicy batchPolicy_;
  BatchTransformFunction<T> batchFn_;

  int64_t preBatchSize_; // Size of the dataset before batching
  int64_t size_;
};

} // namespace objdet
} // namespace app
} // namespace fl
