/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/common/DevicePtr.h"

#include <af/internal.h>

namespace fl {

DevicePtr::DevicePtr(const af::array& in) : arr_(in.get()) {
  if (in.isempty()) {
    ptr_ = nullptr;
  } else {
    if (!af::isLinear(in)) {
      throw std::invalid_argument(
          "can't get device pointer of non-contiguous array");
    }
    ptr_ = in.device<void>();
  }
}

DevicePtr::~DevicePtr() {
  if (ptr_ != nullptr) {
    af_unlock_array(arr_);
  }
}

DevicePtr::DevicePtr(DevicePtr&& d) noexcept : arr_(d.arr_), ptr_(d.ptr_) {
  d.ptr_ = nullptr;
}

DevicePtr& DevicePtr::operator=(DevicePtr&& other) noexcept {
  if (ptr_ != nullptr) {
    af_unlock_array(arr_);
  }
  arr_ = other.arr_;
  ptr_ = other.ptr_;
  other.ptr_ = nullptr;
  return *this;
}

void* DevicePtr::get() const {
  return ptr_;
}

} // namespace fl
