/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <unordered_map>

#include "flashlight/fl/runtime/Device.h"

#include <cuda_runtime.h>

// TODO copied from tensor/CUDAUtils.h to temporarily avoid circular dependency
// merge into this file during final runtime integration
/// usage: `FL_CUDA_CHECK(cudaError_t err[, const char* prefix])`
#define FL_RUNTIME_CUDA_CHECK(...) \
  ::fl::cuda::detail::cudaCheck(__VA_ARGS__, __FILE__, __LINE__)

namespace fl {
namespace cuda {

/**
 * Gets the native id of the active CUDA device.
 *
 * @return the native id of the active CUDA device.
 */
int getActiveDeviceId();

/**
 * Return a mapping from native CUDA device id to available CUDA devices.
 *
 * @return an unordered map from native CUDA device id to CUDA device.
 */
std::unordered_map<int, const std::unique_ptr<Device>> createCUDADevices();

// TODO copied from tensor/CUDAUtils.h to temporarily avoid circular dependency
// merge into this file during final runtime integration
namespace detail {

void cudaCheck(cudaError_t err, const char* file, int line);

void cudaCheck(cudaError_t err, const char* prefix, const char* file, int line);

} // namespace detail

} // namespace cuda
} // namespace fl
