/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/runtime/CUDADevice.h"
#include "flashlight/fl/tensor/CUDAUtils.h"

#include <cuda_runtime.h>

#if FL_USE_ARRAYFIRE
  #include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"
#endif

namespace fl {

CUDADevice::CUDADevice(const int nativeId) : nativeId_(nativeId) {}

int CUDADevice::getNativeId() const {
  return nativeId_;
}

void CUDADevice::setActive() const {
  // We need to go through the backend to ensure its device context consistency.
  // TODO find a better way to do this, e.g., iterate through backends and issue
  // some form of commands to them.
#if FL_USE_ARRAYFIRE && FL_ARRAYFIRE_USE_CUDA
  ArrayFireBackend::getInstance().setDevice(nativeId_);
#endif
  // In case ArrayFire backend is not compiled.
  FL_CUDA_CHECK(cudaSetDevice(nativeId_));
}

} // namespace fl
