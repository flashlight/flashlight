/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Defines.h"

#include <memory>

namespace fl {

class Tensor;

/**
 * DevicePtr provides an RAII wrapper for accessing the device pointer of a
 * Flashlight Tensor array. After calling `device()` on a Flashlight tensor to
 * get a device pointer, its underlying memory is not free until `unlock()` is
 * called - see `fl::Tensor::unlock()`.
 * DevicePtr provides a `std::unique_lock` style API which calls the `unlock()`
 * function in its destructor after getting device pointer. A DevicePtr is
 * movable, but not copyable.
 *
 * Example Usage :
 * \code{.cpp}
 * auto A = Tensor({10, 10});
 * {
 *     DevicePtr devPtr(A); // calls `.device<>()` on array.
 *     void* ptr = devPtr.get();
 * }
 * // devPtr is destructed and A.unlock() is automatically called
 * \endcode
 *
 */
class FL_API DevicePtr {
 public:
  /**
   * Creates a null DevicePtr.
   */
  DevicePtr() : ptr_(nullptr) {}

  /**
   * @param in input array to get device pointer
   */
  explicit DevicePtr(const Tensor& in);

  /**
   *`.unlock()` is called on the underlying array in destructor
   */
  ~DevicePtr();

  DevicePtr(const DevicePtr& other) = delete;

  DevicePtr& operator=(const DevicePtr& other) = delete;

  DevicePtr(DevicePtr&& d) noexcept;

  DevicePtr& operator=(DevicePtr&& other) noexcept;

  bool operator==(const DevicePtr& other) const {
    return get() == other.get();
  }

  void* get() const;

  template <typename T>
  T* getAs() const {
    return reinterpret_cast<T*>(ptr_);
  }

 protected:
  std::unique_ptr<Tensor> tensor_;

 private:
  void* ptr_;
};

struct DevicePtrHasher {
  std::size_t operator()(const DevicePtr& k) const {
    return std::hash<void*>()(k.get());
  }
};

} // namespace fl
