/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <stdexcept>

#include "flashlight/fl/runtime/DeviceType.h"

namespace fl {

/**
 * An abstraction that represents framework-level (as opposed to hardware-level)
 * computing device.
 */
class Device {
 public:
  Device() = default;
  virtual ~Device() = default;

  // no copy/move
  Device(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(const Device&) = delete;
  Device& operator=(Device&&) = delete;

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
    if (T::type != type()) {
      throw std::invalid_argument(
          "[fl::Device::impl] "
          "specified device type doesn't match actual device type.");
    }
    return *(static_cast<const T*>(this));
  }

  // TODO metadata, e.g., device name
  // TODO manage streams on this device
  // TODO sync() which delegates to Stream::sync()
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
