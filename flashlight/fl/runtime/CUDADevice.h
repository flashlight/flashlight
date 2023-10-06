
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/runtime/Device.h"

namespace fl {

/**
 * Represents a CUDA device.
 */
class FL_API CUDADevice : public DeviceTrait<CUDADevice> {
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
  int nativeId() const override;

  /**
   * Set the underlying CUDA device as active.
   */
  void setActiveImpl() const override;
};

} // namespace fl
