/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Halide.h>
#include <HalideBuffer.h>
// TODO: preproc

#include <HalideRuntime.h>
#include <HalideRuntimeCuda.h>

#include <af/array.h>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/DevicePtr.h"

using namespace Halide;

namespace {

/**
 * Gets Halide dims from an ArrayFire array. Halide is column major, so reverse
 * all dimensions.
 */
std::vector<int> getDims(const af::dim4& dims) {
  const auto ndims = dims.ndims();
  std::vector<int> halideDims(ndims);
  for (int i = 0; i < ndims; ++i) {
    halideDims[ndims - 1 - i] = dims.dims[i];
  }
  return halideDims;
}
}

namespace fl {

/**
 * Convert an ArrayFire Array into a Halide Buffer.
 */
template <typename T>
Buffer<T> toHalideBuffer(af::array& arr) {
  T* deviceMem = arr.device<T>(); // TODO: leak, fixme
  Buffer<T> buffer(getDims(arr.dims()));
  // Target is CUDA only
  const Target target = get_jit_target_from_environment().with_feature(Target::Feature::CUDA);
  const DeviceAPI deviceApi = DeviceAPI::CUDA;
  buffer.device_wrap_native(deviceApi, (uint64_t)deviceMem, target);
  buffer.set_host_dirty();
  // buffer.device_wrap_native(
  //     halide_cuda_device_interface(), (uint64_t)deviceMem);
  return buffer;
}

/**
 * Convert an Flashlight Variable into a Halide Buffer.
 */
template <typename T>
Buffer<T> toHalideBuffer(fl::Variable& var) {
  return toHalideBuffer<T>(var.array());
}
}
