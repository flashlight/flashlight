/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>
#include <utility>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

DevicePtr::DevicePtr(const Tensor& in)
    : tensor_(std::make_unique<Tensor>(in.shallowCopy())) {
  if (tensor_->isEmpty()) {
    ptr_ = nullptr;
  } else {
    if (!tensor_->isContiguous()) {
      throw std::invalid_argument(
          "can't get device pointer of non-contiguous Tensor");
    }
    ptr_ = tensor_->device<void>();
  }
}

DevicePtr::~DevicePtr() {
  if (ptr_ != nullptr) {
    tensor_->unlock();
  }
}

DevicePtr::DevicePtr(DevicePtr&& d) noexcept
    : tensor_(std::move(d.tensor_)), ptr_(d.ptr_) {
  d.ptr_ = nullptr;
}

DevicePtr& DevicePtr::operator=(DevicePtr&& other) noexcept {
  if (ptr_ != nullptr) {
    tensor_->unlock();
  }
  tensor_ = std::move(other.tensor_);
  ptr_ = other.ptr_;
  other.ptr_ = nullptr;
  return *this;
}

void* DevicePtr::get() const {
  return ptr_;
}

} // namespace fl
