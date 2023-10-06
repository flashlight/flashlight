/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/dataset/BatchDataset.h"

#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace fl {
BatchDataset::BatchDataset(
    std::shared_ptr<const Dataset> dataset,
    int64_t batchsize,
    BatchDatasetPolicy policy /* = BatchDatasetPolicy::INCLUDE_LAST */,
    const std::vector<BatchFunction>& batchfns /* = {} */)
    : dataset_(dataset),
      batchSize_(batchsize),
      batchPolicy_(policy),
      batchFns_(batchfns) {
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

BatchDataset::BatchDataset(
    std::shared_ptr<const Dataset> dataset,
    const std::vector<int64_t>& batchSizes,
    const std::vector<BatchFunction>& batchfns /* = {} */)
    : dataset_(dataset), cumSumBatchSize_(batchSizes), batchFns_(batchfns) {
  if (!dataset_) {
    throw std::invalid_argument("dataset to be batched is null");
  }
  if (cumSumBatchSize_.empty()) {
    throw std::invalid_argument("batch size vector should not be empty");
  }
  std::partial_sum(
      cumSumBatchSize_.begin(),
      cumSumBatchSize_.end(),
      cumSumBatchSize_.begin());
  preBatchSize_ = dataset_->size();
  size_ = cumSumBatchSize_.size();
}

std::vector<Tensor> BatchDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  int64_t start, end;
  if (cumSumBatchSize_.empty()) {
    // batchsize is given
    start = batchSize_ * idx;
    end = std::min(start + batchSize_, preBatchSize_);
  } else {
    // specific batchsizes array is provided
    start = idx == 0 ? 0 : cumSumBatchSize_[idx - 1];
    end = std::min(cumSumBatchSize_[idx], preBatchSize_);
  }
  return makeBatchFromRange(dataset_, batchFns_, start, end);
}

int64_t BatchDataset::size() const {
  return size_;
}
} // namespace fl
