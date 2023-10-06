/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/runtime/DeviceType.h"
#include "flashlight/fl/runtime/Stream.h"

namespace fl {

// throw invalid_argument with descriptive message if given types don't match
FL_API void deviceImplTypeCheck(DeviceType expect, DeviceType actual);

/**
 * An abstraction that represents framework-level (as opposed to hardware-level)
 * computing device.
 */
class FL_API Device {
  std::unordered_set<std::shared_ptr<Stream>> streams_;
  // Used to update internal backend state for active device, thereby
  // eliminating the `setActive --> AnyTensorBackendImpl` dependency(s).
  std::vector<std::function<void(int)>> setActiveCallbacks_;

 protected:
  /**
   * Set this device as the active device, without worrying about the callbacks.
   */
  virtual void setActiveImpl() const = 0;

 public:
  Device() = default;
  virtual ~Device() = default;

  // no copy/move
  Device(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(const Device&) = delete;
  Device& operator=(Device&&) = delete;

  /**
   * Return all streams managed by this device.
   *
   * @return an immutable vector reference containing all streams managed by
   * this device.
   */
  virtual const std::unordered_set<std::shared_ptr<Stream>>& getStreams() const;

  /**
   * Let this device manage given stream. Do nothing if it was already added.
   *
   * Throws runtime_error if stream is owned by a different device than this
   * one.
   */
  virtual void addStream(std::shared_ptr<Stream> stream);

  /**
   * Block calling thread and synchronize w.r.t. all streams on this device.
   */
  virtual void sync() const;

  /**
   * Get the native ID of this device (semantics are implementation-dependent).
   *
   * @return the native ID of this device.
   */
  virtual int nativeId() const = 0;

  /**
   * Returns the type of this device.
   *
   * @return a enum denoting device type.
   */
  virtual DeviceType type() const = 0;

  /**
   * Set this device as the active device and invokes any callbacks added.
   */
  void setActive() const;

  /**
   * Lets this device keep track of the given callback (along with previously
   * added ones), which will be invoked with the device's native ID after
   * setting the device active.
   *
   * @param[in] callback the callback to be invoked with this device's native ID
   */
  void addSetActiveCallback(std::function<void(int)> callback);

  /**
   * Get the underlying implementation of this device.
   *
   * Throws invalid_argument if the specified type does not match the actual
   * derived device type.
   *
   * @return an immutable reference to the specified device type.
   */
  template <typename T>
  const T& impl() const {
    deviceImplTypeCheck(T::type, type());
    return *(static_cast<const T*>(this));
  }

  /**
   * Get the underlying implementation of this device.
   *
   * Throws invalid_argument if the specified type does not match the actual
   * derived device type.
   *
   * @return a reference to the specified device type.
   */
  template <typename T>
  T& impl() {
    deviceImplTypeCheck(T::type, type());
    return *(static_cast<T*>(this));
  }

  // TODO metadata, e.g., device name
};

/**
 * A trait for some generic device functionalities.
 *
 * REQUIRED definition in derived class:
 *   static DeviceType type;
 */
template <typename Derived>
class DeviceTrait : public Device {
 public:
  DeviceType type() const override {
    return Derived::type;
  }
};

/**
 * A dummy to represent CPU device.
 */
class FL_API X64Device : public DeviceTrait<X64Device> {
 public:
  static constexpr DeviceType type = DeviceType::x64;

  X64Device() = default;
  int nativeId() const override;
  void setActiveImpl() const override;
};

} // namespace fl
