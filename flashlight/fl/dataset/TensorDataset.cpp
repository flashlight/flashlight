/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/dataset/TensorDataset.h"

#include <array>
#include <stdexcept>

#include "flashlight/fl/tensor/Index.h"

namespace fl {

TensorDataset::TensorDataset(const std::vector<Tensor>& dataTensors)
    : dataTensors_(dataTensors), size_(0) {
  if (dataTensors_.empty()) {
    throw std::invalid_argument("no tensors passed to TensorDataset");
  }

  for (const auto& tensor : dataTensors_) {
    auto ndims = tensor.ndim();
    if (ndims == 0) {
      throw std::invalid_argument("tensor for TensorDataset can't be empty");
    }

    auto lastdim = ndims - 1;
    int64_t cursz = tensor.dim(lastdim);
    size_ = std::max(size_, cursz);
  }
}

std::vector<Tensor> TensorDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  std::vector<Tensor> result(dataTensors_.size());
  for (int64_t i = 0; i < dataTensors_.size(); ++i) {
    auto& tensor = dataTensors_[i];

    std::vector<fl::Index> sel(tensor.ndim(), fl::span);
    auto lastdim = tensor.ndim() - 1;
    if (idx < tensor.dim(lastdim)) {
      sel[lastdim] = idx;
      result[i] = tensor(sel);
    }
  }
  return result;
}

int64_t TensorDataset::size() const {
  return size_;
}
} // namespace fl
