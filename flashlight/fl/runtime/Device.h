/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "flashlight/fl/runtime/DeviceType.h"
#include "flashlight/fl/runtime/Stream.h"

namespace fl {

// throw invalid_argument with descriptive message if given types don't match
void deviceImplTypeCheck(DeviceType expect, DeviceType actual);

/**
 * An abstraction that represents framework-level (as opposed to hardware-level)
 * computing device.
 */
class Device {
  std::unordered_set<std::shared_ptr<runtime::Stream>> streams_;

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
  virtual const std::unordered_set<std::shared_ptr<runtime::Stream>>&
    getStreams() const;

  /**
   * Let this device manage given stream. Do nothing if it was already added.
   *
   * Throws runtime_error if stream is owned by a different device than this
   * one.
   */
  virtual void addStream(std::shared_ptr<runtime::Stream> stream);

  /**
   * Synchronize w.r.t. all streams on this device.
   *
   * @return a future representing the completion of all streams on this device.
   */
  virtual std::future<void> sync() const;

  /**
   * Returns the type of this device.
   *
   * @return a enum denoting device type.
   */
  virtual DeviceType type() const = 0;

  /**
   * Set this device as the active device.
   */
  virtual void setActive() const = 0;

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
class X64Device : public DeviceTrait<X64Device> {
 public:
  static constexpr DeviceType type = DeviceType::x64;

  X64Device() = default;
  void setActive() const override;
};

} // namespace fl
