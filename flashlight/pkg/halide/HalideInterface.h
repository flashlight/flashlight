/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdexcept>
#include <vector>

#include <Halide.h>
#include <HalideBuffer.h>
#include <HalideRuntime.h>
#include <HalideRuntimeCuda.h>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/Types.h"

#define FL_HALIDE_CHECK(exp)                                       \
  if (exp) {                                                       \
    throw std::runtime_error(                                      \
        "Halide exp " + std::string(#exp) + " failed with code " + \
        std::to_string(exp));                                      \
  }

namespace fl {
namespace pkg {
namespace halide {

/**
 * Gets Halide dims from an ArrayFire array. Halide is row major, so reverse
 * all dimensions.
 */
std::vector<int> flToHalideDims(const Shape& dims);

/**
 * Gets Halide dims from an ArrayFire array. Halide is column major, so reverse
 * all dimensions.
 */
Shape halideToFlDims(const Halide::Buffer<void>& buffer);

fl::dtype halideRuntimeTypeToFlType(halide_type_t type);

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
  HalideBufferWrapper(Tensor& tensor) {
    if (tensor.backendType() != TensorBackendType::ArrayFire) {
      throw std::runtime_error(
          "[HalideBufferWrapper] Only support Tensor with ArrayFireBackend");
    }
    devicePtr_ = DevicePtr(tensor);
    halideBuffer_ = Halide::Buffer<T>(flToHalideDims(tensor.shape()));
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
 * USE WITH CAUTION: Convert a Flashlight Tensor into a Halide Buffer. The
 * resulting Halide buffer's memory will be **unmanaged** - the underlying array
 * will need to be unlocked with `fl::Tensor::unlock()` else memory will leak.
 */
template <typename T>
Halide::Buffer<T> toHalideBuffer(Tensor& arr) {
  if (arr.backendType() != TensorBackendType::ArrayFire) {
    throw std::runtime_error(
        "[HalideBufferWrapper] Only support Tensor with ArrayFireBackend");
  }
  // Since the buffer manages the memory, give it a persistent pointer that
  // won't be unlocked or invalidated if the Array falls out of scope.
  void* deviceMem = arr.device<void>();
  Halide::Buffer<T> buffer(flToHalideDims(arr.shape()));
  // Target is CUDA only -- TODO: change based on location of Tensor
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
 * @return a Flashlight Tensor that has the same underlying memory and
 * dimensions as the Halide Buffer.
 */
template <typename T>
Tensor fromHalideBuffer(Halide::Buffer<T>& buffer) {
  T* deviceMem = reinterpret_cast<T*>(buffer.raw_buffer()->device);
  if (buffer.get()->device_ownership() ==
      Halide::Runtime::BufferDeviceOwnership::WrappedNative) {
    FL_HALIDE_CHECK(buffer.device_detach_native());
  } else {
    throw std::invalid_argument(
        "fl::pkg::runtime::fromHalideBuffer can only be called with buffers created "
        "with fl::pkg::runtime::toHalideBuffer or buffers that have unmanaged buffer "
        "device ownership policies.");
  }
  return Tensor::fromBuffer(
      halideToFlDims(buffer), deviceMem, MemoryLocation::Device);
}

} // namespace detail

} // namespace halide
} // namespace pkg
} // namespace fl
