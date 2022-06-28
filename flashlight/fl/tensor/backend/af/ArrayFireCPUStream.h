/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/runtime/Stream.h"

namespace fl {

/**
 * An abstraction for ArrayFire's CPU Stream with controlled creation methods.
 */
class ArrayFireCPUStream : public runtime::StreamTrait<ArrayFireCPUStream> {
  X64Device& device_{DeviceManager::getInstance().getActiveDevice(DeviceType::x64).impl<X64Device>()};

 public:
  // prevent name hiding
  using StreamTrait<ArrayFireCPUStream>::relativeSync;

  static constexpr runtime::StreamType type = runtime::StreamType::Synchronous;

  /**
   * Creates an ArrayFireCPUStream and automatically register it with
   * the active x64 device from DeviceManager.
   *
   * @return a shared pointer to the created ArrayFireCPUStream.
   */
  static std::shared_ptr<ArrayFireCPUStream> create();

  X64Device& device() override;
  const X64Device& device() const override;
  std::future<void> sync() const override;
  void relativeSync(const ArrayFireCPUStream& waitOn) const override;
};

} // namespace fl
