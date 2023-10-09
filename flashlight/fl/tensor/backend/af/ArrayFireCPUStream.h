/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/runtime/SynchronousStream.h"

namespace fl {

/**
 * An abstraction for ArrayFire's CPU Stream with controlled creation methods.
 */
class ArrayFireCPUStream : public SynchronousStream {
 public:
  /**
   * Creates an ArrayFireCPUStream and automatically register it with
   * the active x64 device from DeviceManager.
   *
   * @return a shared pointer to the created ArrayFireCPUStream.
   */
  static std::shared_ptr<ArrayFireCPUStream> create();

  void sync() const override;
};

} // namespace fl
