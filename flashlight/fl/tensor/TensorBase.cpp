/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorBase.h"

#include <utility>

#include "flashlight/fl/tensor/TensorAdapter.h"

namespace fl {

Tensor::Tensor(std::unique_ptr<TensorAdapterBase> adapter)
    : impl_(std::move(adapter)) {}

Tensor::~Tensor() {}

Shape Tensor::shape() const {
  return impl_->shape();
}

dtype Tensor::type() const {
  return impl_->type();
}

Tensor Tensor::astype(const dtype type) {
  return impl_->astype(type);
}

TensorBackend Tensor::backend() const {
  return impl_->backend();
}

} // namespace fl
