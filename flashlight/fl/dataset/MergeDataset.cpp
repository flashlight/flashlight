/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/dataset/MergeDataset.h"

namespace fl {
MergeDataset::MergeDataset(
    const std::vector<std::shared_ptr<const Dataset>>& datasets)
    : datasets_(datasets) {
  size_ = 0;
  for (auto dataset : datasets_) {
    size_ = std::max(dataset->size(), size_);
  }
}

std::vector<af::array> MergeDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);

  std::vector<af::array> result;
  for (auto dataset : datasets_) {
    if (idx < dataset->size()) {
      auto f = dataset->get(idx);
      result.insert(
          result.end(),
          std::make_move_iterator(f.begin()),
          std::make_move_iterator(f.end()));
    }
  }
  return result;
}

int64_t MergeDataset::size() const {
  return size_;
}
} // namespace fl
