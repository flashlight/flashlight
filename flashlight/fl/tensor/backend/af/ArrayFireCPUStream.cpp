/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireCPUStream.h"

#include <af/device.h>

#include "flashlight/fl/runtime/DeviceManager.h"

namespace fl {

X64Device& ArrayFireCPUStream::device() {
  return device_;
}

const X64Device& ArrayFireCPUStream::device() const {
  return device_;
}

std::shared_ptr<ArrayFireCPUStream> ArrayFireCPUStream::create() {
  // TODO `std::make_shared` requires a public constructor, which could be
  // abused and lead to unregistered stream. However, it has one internal
  // allocation and is more cache-friendly than `std::shared_ptr`.
  const auto rawStreamPtr = new ArrayFireCPUStream();
  const auto stream = std::shared_ptr<ArrayFireCPUStream>(rawStreamPtr);
  rawStreamPtr->device_.addStream(stream);
  return stream;
}

std::future<void> ArrayFireCPUStream::sync() const {
  // TODO change into blocking
  auto deviceId = af::getDevice();
  return std::async(std::launch::async, [deviceId]{
    af::sync(deviceId);
  });
}

void ArrayFireCPUStream::relativeSync(const ArrayFireCPUStream& waitOn) const {
  waitOn.sync().wait();
}

} // namespace fl
