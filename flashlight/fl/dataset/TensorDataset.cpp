/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <stdexcept>

#include "flashlight/fl/dataset/TensorDataset.h"

namespace fl {

TensorDataset::TensorDataset(const std::vector<af::array>& datatensors)
    : dataTensors_(datatensors), size_(0) {
  if (dataTensors_.empty()) {
    throw std::invalid_argument("no tensors passed to TensorDataset");
  }
  for (const auto& tensor : dataTensors_) {
    auto ndims = tensor.numdims();
    if (ndims == 0) {
      throw std::invalid_argument("tensor for TensorDataset can't be empty");
    }
    auto lastdim = ndims - 1;
    int64_t cursz = tensor.dims(lastdim);
    size_ = std::max(size_, cursz);
  }
}

std::vector<af::array> TensorDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  std::vector<af::array> result(dataTensors_.size());
  for (int64_t i = 0; i < dataTensors_.size(); ++i) {
    std::array<af::seq, 4> sel{af::span, af::span, af::span, af::span};
    auto& tensor = dataTensors_[i];
    auto lastdim = tensor.numdims() - 1;
    if (idx < tensor.dims(lastdim)) {
      sel[lastdim] = af::seq(idx, idx);
      result[i] = tensor(sel[0], sel[1], sel[2], sel[3]);
    }
  }
  return result;
}

int64_t TensorDataset::size() const {
  return size_;
}
} // namespace fl
