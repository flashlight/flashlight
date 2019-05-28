/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>

namespace fl {

/**
 * DevicePtr provides a wrapper for accessing device pointer of an Arrayfire
 * array in a safer way.  After calling `device()` on arrayfire array to get
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
   * @param in input array to get device pointer
   */
  explicit DevicePtr(const af::array& in);

  /**
   * Move construction is allowed
   */
  DevicePtr(DevicePtr&& d) = default;

  /**
   * Move assignment is allowed
   */
  DevicePtr& operator=(DevicePtr&& other) = default;

  /**
   * Get device pointer of the array
   */
  void* get() const;

  /**
   * Copy constructor is deleted
   */
  DevicePtr(const DevicePtr& other) = delete;

  /**
    Copy assignment operator is deleted
  */
  DevicePtr& operator=(const DevicePtr& other) = delete;

  bool operator==(const DevicePtr& other) const {
    return get() == other.get();
  }

  /**
   *`.unlock()` is called on the underlying array in destructor
   */
  ~DevicePtr();

 private:
  const af::array* arr_;
  void* ptr_;
};

struct DevicePtrHasher {
  std::size_t operator()(const DevicePtr& k) const {
    return std::hash<void*>()(k.get());
  }
};

} // namespace fl
