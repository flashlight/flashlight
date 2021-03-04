/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdexcept>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <Halide.h>
#include <HalideBuffer.h>
#include <HalideRuntime.h>
#include <HalideRuntimeCuda.h>

#include <af/array.h>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/DevicePtr.h"

#define FL_HALIDE_CHECK(exp)                                       \
  if (exp) {                                                       \
    throw std::runtime_error(                                      \
        "Halide exp " + std::string(#exp) + " failed with code " + \
        std::to_string(exp));                                      \
  }

namespace fl {
namespace ext {

/**
 * Gets Halide dims from an ArrayFire array. Halide is row major, so reverse
 * all dimensions.
 */
std::vector<int> afToHalideDims(const af::dim4& dims);

/**
 * Gets Halide dims from an ArrayFire array. Halide is column major, so reverse
 * all dimensions.
 */
af::dim4 halideToAfDims(const Halide::Buffer<void>& buffer);

af::dtype halideRuntimeTypeToAfType(halide_type_t type);

/**
 * A thin wrapper around an ArrayFire array as converted to a Halide buffer.
 *
 * Uses RAII via DevicePtr to ensure that the memory associated with the
 * underlying Array is properly managed as it relates to the lifetime of hte
 * Halide Buffer.
 *
 * The toHalideBuffer and fromHalideBuffer functions provide indefinite lifetime
 * guarantees around their conversions which are unmanaged and require manual
 * cleanup. Use this class instead for automatic lifetime management.
 */
template <typename T>
class HalideBufferWrapper {
 public:
  HalideBufferWrapper(af::array& array) {
    devicePtr_ = DevicePtr(array);
    halideBuffer_ = Halide::Buffer<T>(afToHalideDims(array.dims()));
    // Halide::Buffer::device_detach_native(...) is implicitly called by the
    // Halide::Buffer dtor which will preserve the Array's underlying memory
    FL_HALIDE_CHECK(halideBuffer_.device_wrap_native(
        halide_cuda_device_interface(), (uint64_t)devicePtr_.get()));
    halideBuffer_.set_device_dirty();
  }

  Halide::Buffer<T>& getBuffer() {
    return halideBuffer_;
  }

  Halide::Runtime::Buffer<T>& getRuntimeBuffer() {
    return *halideBuffer_.get();
  }

 private:
  DevicePtr devicePtr_;
  Halide::Buffer<T> halideBuffer_;
};

namespace detail {

/**
 * You probably want to use HalideBufferWrapper rather than calling this
 * function manually.
 *
 * USE WITH CAUTION: Convert an ArrayFire Array into a Halide Buffer. The
 * resulting Halide buffer's memory will be **unmanaged** - the underlying array
 * will need to be unlocked with `af_unlock_array` else memory will leak.
 */
template <typename T>
Halide::Buffer<T> toHalideBuffer(af::array& arr) {
  // Since the buffer manages the memory, give it a persistent pointer that
  // won't be unlocked or invalidated if the Array falls out of scope.
  void* deviceMem = arr.device<void>();
  Halide::Buffer<T> buffer(afToHalideDims(arr.dims()));
  // Target is CUDA only -- TODO: change based on location of af::array
  // and try to move away from halide_cuda_device_interface()
  // const Halide::Target target =
  //     Halide::get_target_from_environment().with_feature(
  //         Halide::Target::Feature::CUDA);
  // const Halide::DeviceAPI deviceApi = Halide::DeviceAPI::CUDA;
  // int err = buffer.device_wrap_native(deviceApi, (uint64_t)deviceMem,
  // target);
  FL_HALIDE_CHECK(buffer.device_wrap_native(
      halide_cuda_device_interface(), (uint64_t)deviceMem));
  buffer.set_device_dirty();
  return buffer;
}

/**
 * You probably want to use HalideBufferWrapper rather than calling this
 * function manually.
 *
 * USE WITH CAUTION: convert a Halide Buffer into an ArrayFire Array. Grabs the
 * Halide Buffer's underlying memory and creates a new Array with it. Only
 * buffer types created with Halide::BufferDeviceOwnership::Unmanaged can be
 * convered since otherwise the underlying memory will be freed once the Buffer
 * is destroyed.
 *
 * @param buffer the Halide buffer with which to create the Array
 * @return an af::array that has the same underlying memory and diensions as the
 * Halide Buffer.
 */
template <typename T>
af::array fromHalideBuffer(Halide::Buffer<T>& buffer) {
  T* deviceMem = reinterpret_cast<T*>(buffer.raw_buffer()->device);
  if (buffer.get()->device_ownership() ==
      Halide::Runtime::BufferDeviceOwnership::WrappedNative) {
    FL_HALIDE_CHECK(buffer.device_detach_native());
  } else {
    throw std::invalid_argument(
        "fl::ext::fromHalideBuffer can only be called with buffers created "
        "with fl::ext::toHalideBuffer or buffers that have unmanaged buffer "
        "device ownership policies.");
  }
  return af::array(halideToAfDims(buffer), deviceMem, afDevice);
}
} // namespace detail
} // namespace ext
} // namespace fl
