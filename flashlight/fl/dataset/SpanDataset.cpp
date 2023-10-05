/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/dataset/SpanDataset.h"

namespace fl {
SpanDataset::SpanDataset(
    std::shared_ptr<const Dataset> dataset,
    const int64_t offset,
    const int64_t length)
    : dataset_(dataset), offset_(offset) {
  size_ = (length < 0) ? (dataset_->size() - offset_) : length;
  if (size_ + offset_ > dataset_->size()) {
    throw std::out_of_range(
        "Dataset length out of range (larger than underlying dataset)");
  }
}

std::vector<Tensor> SpanDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);

  std::vector<Tensor> result;
  auto f = dataset_->get(idx + offset_);
  result.insert(
      result.end(),
      std::make_move_iterator(f.begin()),
      std::make_move_iterator(f.end()));
  return result;
}

int64_t SpanDataset::size() const {
  return size_;
}
} // namespace fl
