/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>

#include "flashlight/fl/runtime/Device.h"
#include "flashlight/fl/runtime/DeviceManager.h"

namespace fl {

void deviceImplTypeCheck(DeviceType expect, DeviceType actual) {
  if (expect != actual) {
    std::ostringstream oss;
    oss << "[fl::Device::impl] "
        << "specified device type: [" << expect << "] "
        << "doesn't match actual device type: [" << actual << "]";
    throw std::invalid_argument(oss.str());
  }
}

const std::unordered_set<std::shared_ptr<Stream>>& Device::getStreams() const {
  return streams_;
}

void Device::addStream(std::shared_ptr<Stream> stream) {
  if (&stream->device() != this) {
    throw std::runtime_error(
      "[Device::addStream] Must add stream to owner device");
  }
  streams_.insert(stream);
}

void Device::sync() const {
  for (const auto& stream : streams_) {
    stream->sync();
  }
}

void Device::addSetActiveCallback(std::function<void(int)> callback) {
  setActiveCallbacks_.push_back(std::move(callback));
}

void Device::setActive() const {
  setActiveImpl();
  for (auto& callback : setActiveCallbacks_) {
    callback(nativeId());
  }
}

int X64Device::nativeId() const {
  return fl::kX64DeviceId;
}

void X64Device::setActiveImpl() const {
  // no op, CPU device is always active
}

} // namespace fl
