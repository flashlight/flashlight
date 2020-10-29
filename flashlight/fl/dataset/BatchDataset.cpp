/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/dataset/BatchDataset.h"

#include <math.h>
#include <array>
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
      size_ = ceil(static_cast<double>(preBatchSize_) / batchSize_);
      break;
    case BatchDatasetPolicy::SKIP_LAST:
      size_ = floor(static_cast<double>(preBatchSize_) / batchSize_);
      break;
    case BatchDatasetPolicy::DIVISIBLE_ONLY:
      if (size_ % batchSize_ != 0) {
        throw std::invalid_argument(
            "dataset is not evenly divisible into batches");
      }
      size_ = ceil(static_cast<double>(preBatchSize_) / batchSize_);
      break;
    default:
      throw std::invalid_argument("unknown BatchDatasetPolicy");
  }
}

std::vector<af::array> BatchDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);

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
  std::vector<af::array> result(buffer.size());
  for (int64_t i = 0; i < buffer.size(); ++i) {
    result[i] =
        makeBatch(buffer[i], (i < batchFns_.size()) ? batchFns_[i] : nullptr);
  }
  return result;
}

af::array BatchDataset::makeBatch(
    const std::vector<af::array>& data,
    const BatchFunction& batchFn) const {
  if (batchFn) {
    return batchFn(data);
  }
  // Using default batching function
  if (data.empty()) {
    return af::array();
  }
  auto dims = data[0].dims();

  for (const auto& d : data) {
    if (d.dims() != dims) {
      throw std::invalid_argument("dimension mismatch while batching dataset");
    }
  }

  int ndims = (data[0].elements() > 1) ? dims.ndims() : 0;

  if (ndims >= 4) {
    throw std::invalid_argument("# of dims must be < 4 for batching");
  }
  dims[ndims] = data.size();
  auto batcharr = af::array(dims, data[0].type());

  for (size_t i = 0; i < data.size(); ++i) {
    std::array<af::seq, 4> sel{af::span, af::span, af::span, af::span};
    sel[ndims] = af::seq(i, i);
    batcharr(sel[0], sel[1], sel[2], sel[3]) = data[i];
  }
  return batcharr;
}

int64_t BatchDataset::size() const {
  return size_;
}
} // namespace fl
