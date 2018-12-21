/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "DevicePtr.h"

#include "Utils.h"

namespace fl {

DevicePtr::DevicePtr(const af::array& in) : arr_(&in) {
  if (arr_->isempty()) {
    ptr_ = nullptr;
  } else {
    detail::assertLinear(in);
    ptr_ = arr_->device<void>();
  }
}

void* DevicePtr::get() const {
  return ptr_;
}

DevicePtr::~DevicePtr() {
  if (ptr_ != nullptr) {
    arr_->unlock();
  }
}

} // namespace fl
