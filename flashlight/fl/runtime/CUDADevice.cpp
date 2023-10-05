/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/CUDADevice.h"
#include "flashlight/fl/runtime/CUDAUtils.h"

namespace fl {

CUDADevice::CUDADevice(const int nativeId) : nativeId_(nativeId) {}

int CUDADevice::nativeId() const {
  return nativeId_;
}

void CUDADevice::setActiveImpl() const {
  FL_CUDA_CHECK(cudaSetDevice(nativeId_));
}

} // namespace fl
