/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/dataset/TransformDataset.h"

namespace fl {

TransformDataset::TransformDataset(
    std::shared_ptr<const Dataset> dataset,
    const std::vector<TransformFunction>& transformfns)
    : dataset_(dataset), transformFns_(transformfns) {
  if (!dataset_) {
    throw std::invalid_argument("dataset to be transformed is null");
  }
}

std::vector<af::array> TransformDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);

  auto result = dataset_->get(idx);

  for (int64_t i = 0; i < result.size(); ++i) {
    if (i >= transformFns_.size() || !transformFns_[i]) {
      continue;
    }
    result[i] = transformFns_[i](result[i]);
  }
  return result;
}

int64_t TransformDataset::size() const {
  return dataset_->size();
}
} // namespace fl
