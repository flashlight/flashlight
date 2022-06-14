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

} // namespace cuda
} // namespace fl
