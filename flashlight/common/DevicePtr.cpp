/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "DevicePtr.h"

#include <af/internal.h>

namespace fl {

DevicePtr::DevicePtr(const af::array& in) : arr_(&in) {
  if (arr_->isempty()) {
    ptr_ = nullptr;
  } else {
    if (!af::isLinear(in)) {
      throw std::invalid_argument(
          "can't get device pointer of non-contiguous array");
    }
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
