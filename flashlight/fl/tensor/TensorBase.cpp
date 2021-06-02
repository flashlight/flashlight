/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorBase.h"

#include <utility>

#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/TensorBackend.h"

namespace fl {

Tensor::Tensor(std::unique_ptr<TensorAdapterBase> adapter)
    : impl_(std::move(adapter)) {}

Tensor::~Tensor() {}

Tensor::Tensor() : impl_(detail::getDefaultAdapter()) {}

const Shape& Tensor::shape() const {
  return impl_->shape();
}

dtype Tensor::type() const {
  return impl_->type();
}

Tensor Tensor::astype(const dtype type) {
  return impl_->astype(type);
}

TensorBackendType Tensor::backendType() const {
  return impl_->backendType();
}

TensorBackend& Tensor::backend() const {
  return impl_->backend();
}

/* --------------------------- Tensor Operators --------------------------- */

/************************** Unary Operators ***************************/
Tensor exp(const Tensor& tensor) {
  return tensor.backend().exp(tensor);
}

Tensor log(const Tensor& tensor) {
  return tensor.backend().log(tensor);
}

} // namespace fl
