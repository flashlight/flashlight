
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/runtime/Device.h"

namespace fl {

/**
 * Represents a CUDA device.
 */
class CUDADevice : public DeviceTrait<CUDADevice> {
  // native ID of the underlying CUDA device
  const int nativeId_;
  // TODO metadata, e.g., memory/compute capacity

 public:
  static constexpr DeviceType type = DeviceType::CUDA;

  /**
   * Creates a wrapper around the CUDA device with given native device ID.
   *
   * @param[in] nativeId the CUDA device ID with which to create this Device.
   */
  explicit CUDADevice(int nativeId);

  /**
   * Returns the native CUDA device ID.
   *
   * @return an integer representing the native CUDA device ID.
   */
  int getNativeId() const;

  /**
   * Set the underlying CUDA device as active and ensure device context
   * consistency in applicable backends.
   */
  void setActive() const override;
};

} // namespace fl
