/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/common/CudaUtils.h"

#include <sstream>

#include <af/device.h>

namespace fl {
namespace cuda {

cudaStream_t getActiveStream() {
  auto af_id = af::getDevice();
  return afcu::getStream(af_id);
}

namespace detail {

void check(cudaError_t err, const char* file, int line) {
  check(err, "", file, line);
}

void check(cudaError_t err, const char* prefix, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream ess;
    ess << prefix << '[' << file << ':' << line
       << "] CUDA error: " << cudaGetErrorString(err);
    throw std::runtime_error(ess.str());
  }
}
} // namespace detail

} // namespace cuda
} // namespace fl
