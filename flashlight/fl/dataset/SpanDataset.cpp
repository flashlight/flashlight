/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/dataset/SpanDataset.h"

namespace fl {
SpanDataset::SpanDataset(
    std::shared_ptr<const Dataset> dataset,
    const int64_t offset,
    const int64_t size)
    : dataset_(dataset), offset_(offset) {
  size_ = (size < 0) ? (dataset_->size() - offset_) : size;
  if (size_ + offset_ > dataset_->size()) {
    throw std::out_of_range(
        "Specified size out of range (larger than underlying dataset)");
  }
}

std::vector<Tensor> SpanDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);

  return dataset_->get(idx + offset_);
}

int64_t SpanDataset::size() const {
  return size_;
}
} // namespace fl
