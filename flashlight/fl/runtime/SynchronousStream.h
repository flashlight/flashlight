/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/runtime/Stream.h"

namespace fl {

/**
 * An abstraction for a synchronous stream. The word "synchronous" describes the
 * relative synchronization strategy, i.e., it merely delegates to `sync`.
 */
class FL_API SynchronousStream : public StreamTrait<SynchronousStream> {
 protected:
  X64Device& device_{DeviceManager::getInstance()
                         .getActiveDevice(DeviceType::x64)
                         .impl<X64Device>()};

 public:
  // prevent name hiding
  using StreamTrait<SynchronousStream>::relativeSync;

  static constexpr StreamType type = StreamType::Synchronous;

  X64Device& device() override;
  const X64Device& device() const override;
  void relativeSync(const SynchronousStream& waitOn) const override;
};

} // namespace fl
