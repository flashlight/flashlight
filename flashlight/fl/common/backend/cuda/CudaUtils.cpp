/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/backend/cuda/CudaUtils.h"

#include <sstream>
#include <stdexcept>

#include <af/device.h>

namespace fl {
namespace cuda {

cudaStream_t getActiveStream() {
  auto af_id = af::getDevice();
  return afcu::getStream(af_id);
}

void synchronizeStreams(
    cudaStream_t blockee,
    cudaStream_t blockOn,
    cudaEvent_t event) {
  FL_CUDA_CHECK(cudaEventRecord(event, blockOn));
  FL_CUDA_CHECK(cudaStreamWaitEvent(blockee, event, 0));
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
