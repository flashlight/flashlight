/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/runtime/CUDADevice.h"
#include "flashlight/fl/runtime/CUDAUtils.h"

#include <cuda_runtime.h>

namespace fl::cuda {

int getActiveDeviceId() {
  int cudaActiveDeviceId = 0;
  FL_CUDA_CHECK(cudaGetDevice(&cudaActiveDeviceId));
  return cudaActiveDeviceId;
}

std::unordered_map<int, const std::unique_ptr<Device>> createCUDADevices() {
  std::unordered_map<int, const std::unique_ptr<Device>> idToDevice;
  int numCudaDevices = 0;
  FL_CUDA_CHECK(cudaGetDeviceCount(&numCudaDevices));
  for (auto id = 0; id < numCudaDevices; id++) {
    idToDevice.emplace(id, std::make_unique<CUDADevice>(id));
  }
  return idToDevice;
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

} // namespace fl
