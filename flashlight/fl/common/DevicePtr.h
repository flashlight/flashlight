/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>

namespace fl {

/**
 * DevicePtr provides an RAII wrapper for accessing the device pointer of an
 * ArrayFire array. After calling `device()` on arrayfire array to get
 * device pointer, the memory is not free until `unlock()` is called -
 * 'http://arrayfire.org/docs/group__device__func__device.htm'.
 * DevicePtr provides a `std::unique_lock` style API which calls the `unlock()`
 * function in its destructor after getting device pointer. A DevicePtr is
 * movable, but not copyable.
 *
 * Example Usage :
 * \code{.cpp}
 * auto A = af::array(10, 10);
 * {
 *     DevicePtr devPtr(A); // calls `.device<>()` on array.
 *     void* ptr = devPtr.get();
 * }
 * // devPtr is destructed and A.unlock() is automatically called
 * \endcode
 *
 */
class DevicePtr {
 public:
  /**
   * Creates a null DevicePtr.
   */
  DevicePtr() : ptr_(nullptr) {}

  /**
   * @param in input array to get device pointer
   */
  explicit DevicePtr(const af::array& in);

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
  af_array arr_;

 private:
  void* ptr_;
};

struct DevicePtrHasher {
  std::size_t operator()(const DevicePtr& k) const {
    return std::hash<void*>()(k.get());
  }
};

} // namespace fl
