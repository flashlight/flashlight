/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include "TensorDataset.h"

namespace fl {

TensorDataset::TensorDataset(const std::vector<af::array>& datatensors)
    : dataTensors_(datatensors), size_(0) {
  FL_ASSERT(!dataTensors_.empty(), "Empty data tensors map.");
  for (const auto& tensor : dataTensors_) {
    auto lastdim = tensor.numdims() - 1;
    FL_ASSERT(lastdim >= 0, "tensor can't be zero-dimensional.");
    int64_t cursz = tensor.dims(lastdim);
    size_ = std::max(size_, cursz);
  }
}

std::vector<af::array> TensorDataset::get(const int64_t idx) const {
  FL_ASSERT(
      idx >= 0 && idx < size(),
      "Invalid value of idx. idx should be in [0, size())");
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
