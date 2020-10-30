/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/dataset/ConcatDataset.h"

#include <algorithm>
#include <stdexcept>

namespace fl {
ConcatDataset::ConcatDataset(
    const std::vector<std::shared_ptr<const Dataset>>& datasets)
    : datasets_(datasets), size_(0) {
  if (datasets.empty()) {
    throw std::invalid_argument("cannot concat 0 datasets");
  }
  cumulativedatasetsizes_.emplace_back(0);
  for (auto dataset : datasets_) {
    size_ += dataset->size();
    cumulativedatasetsizes_.emplace_back(size_);
  }
}

std::vector<af::array> ConcatDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);

  // get sample from correct dataset
  int64_t datasetidx =
      std::upper_bound(
          cumulativedatasetsizes_.begin(), cumulativedatasetsizes_.end(), idx) -
      cumulativedatasetsizes_.begin() - 1;
  return datasets_[datasetidx]->get(idx - cumulativedatasetsizes_[datasetidx]);
}

int64_t ConcatDataset::size() const {
  return size_;
}
} // namespace fl
