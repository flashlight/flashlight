/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "BatchDataset.h"

#include <math.h>
#include <array>

namespace fl {
BatchDataset::BatchDataset(
    std::shared_ptr<const Dataset> dataset,
    int64_t batchsize,
    BatchDatasetPolicy policy /* = BatchDatasetPolicy::INCLUDE_LAST */,
    PermutationFunction permutationfn /* = nullptr */,
    BatchFunction batchfn /* = nullptr */)
    : dataset_(dataset),
      batchSize_(batchsize),
      batchPolicy_(policy),
      permutationFn_(permutationfn),
      batchFn_(batchfn) {
  FL_ASSERT(dataset_, "Dataset shouldn't be a nullptr");
  FL_ASSERT(batchSize_ > 0, "Batch size must be > 0");
  preBatchSize_ = dataset_->size();
  switch (batchPolicy_) {
    case BatchDatasetPolicy::INCLUDE_LAST:
      size_ = ceil(static_cast<double>(preBatchSize_) / batchSize_);
      break;
    case BatchDatasetPolicy::SKIP_LAST:
      size_ = floor(static_cast<double>(preBatchSize_) / batchSize_);
      break;
    case BatchDatasetPolicy::DIVISIBLE_ONLY:
      FL_ASSERT(
          size_ % batchSize_ == 0,
          "Dataset size must be divisible by batchsize in DIVISIBLE_ONLY mode");
      size_ = ceil(static_cast<double>(preBatchSize_) / batchSize_);
      break;
    default:
      FL_ASSERT(false, "Unexpected policy for BatchDataset");
  }
}

std::vector<af::array> BatchDataset::get(const int64_t idx) const {
  FL_ASSERT(
      idx >= 0 && idx < size(),
      "Invalid value of idx. idx should be in [0, size())");

  std::vector<std::vector<af::array>> buffer;

  int64_t start = batchSize_ * idx;
  int64_t end = std::min(start + batchSize_, preBatchSize_);

  for (int64_t batchidx = start; batchidx < end; ++batchidx) {
    auto fds =
        dataset_->get(permutationFn_ ? permutationFn_(batchidx) : batchidx);
    if (buffer.size() < fds.size()) {
      buffer.resize(fds.size());
    }
    for (int64_t i = 0; i < fds.size(); ++i) {
      buffer[i].emplace_back(fds[i]);
    }
  }
  std::vector<af::array> result(buffer.size());
  for (int64_t i = 0; i < buffer.size(); ++i) {
    result[i] = makeBatch(buffer[i]);
  }
  return result;
}

af::array BatchDataset::makeBatch(const std::vector<af::array>& data) const {
  if (batchFn_) {
    return batchFn_(data);
  }
  // Using default batching function
  if (data.empty()) {
    return af::array();
  }
  auto dims = data[0].dims();

  // assertions
  for (const auto& d : data) {
    FL_ASSERT(d.dims() == dims, "Arrays should have same dims to batch.");
  }

  dim_t ndims = (data[0].elements() > 1) ? dims.ndims() : 0;

  FL_ASSERT(ndims < 4, "Number of dims has to be < 4 for batching.");
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
