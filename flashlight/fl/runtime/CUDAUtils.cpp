/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/runtime/CUDAUtils.h"
// TODO fold this into current file when integrating runtime into Flashlight
#include "flashlight/fl/tensor/CUDAUtils.h"

#include <cuda_runtime.h>

namespace fl {
namespace cuda {

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
    idToDevice.emplace(id, std::make_unique<Device>(DeviceType::CUDA));
  }
  return idToDevice;
}

} // namespace cuda
} // namespace fl
