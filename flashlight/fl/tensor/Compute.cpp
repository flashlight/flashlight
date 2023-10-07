/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Compute.h"

#include <utility>

#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/runtime/DeviceType.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

namespace {

std::unordered_set<const Stream*> tensorsToUniqueStreams(
    const std::vector<Tensor>& tensors) {
  std::unordered_set<const Stream*> uniqueStreams;
  for (const auto& tensor : tensors) {
    uniqueStreams.insert(&tensor.stream());
  }
  return uniqueStreams;
}

std::unordered_set<const Stream*> tensorsToUniqueStreams(
    const std::vector<const Tensor*>& tensors) {
  std::unordered_set<const Stream*> uniqueStreams;
  for (const auto& tensor : tensors) {
    uniqueStreams.insert(&tensor->stream());
  }
  return uniqueStreams;
}

} // namespace

void sync() {
  DeviceManager::getInstance().getActiveDevice(fl::kDefaultDeviceType).sync();
}

void sync(const int deviceId) {
  DeviceManager::getInstance()
      .getDevice(fl::kDefaultDeviceType, deviceId)
      .sync();
}

void sync(const std::unordered_set<DeviceType>& types) {
  const auto& manager = DeviceManager::getInstance();
  // TODO consider launching these `Device::sync` calls non-blockingly
  for (const auto type : types) {
    manager.getActiveDevice(type).sync();
  }
}

void sync(const std::unordered_set<const Device*>& devices) {
  // TODO consider launching these `Device::sync` calls non-blockingly
  for (const auto* device : devices) {
    device->sync();
  }
}

void relativeSync(
    const Stream& wait,
    const std::vector<const Tensor*>& waitOns) {
  // ensure computations are launched
  for (const auto* tensor : waitOns) {
    tensor->backend().eval(*tensor);
  }
  wait.relativeSync(tensorsToUniqueStreams(waitOns));
}

void relativeSync(const Stream& wait, const std::vector<Tensor>& waitOns) {
  // ensure computations are launched
  for (const auto& tensor : waitOns) {
    tensor.backend().eval(tensor);
  }
  wait.relativeSync(tensorsToUniqueStreams(waitOns));
}

void relativeSync(const std::vector<Tensor>& waits, const Stream& waitOn) {
  for (const auto& stream : tensorsToUniqueStreams(waits)) {
    stream->relativeSync(waitOn);
  }
}

void eval(Tensor& tensor) {
  tensor.backend().eval(tensor);
}

int getDevice() {
  return DeviceManager::getInstance()
      .getActiveDevice(fl::kDefaultDeviceType)
      .nativeId();
}

void setDevice(const int deviceId) {
  DeviceManager::getInstance()
      .getDevice(fl::kDefaultDeviceType, deviceId)
      .setActive();
}

int getDeviceCount() {
  return DeviceManager::getInstance().getDeviceCount(fl::kDefaultDeviceType);
}

namespace detail {

void getMemMgrInfo(
    const char* msg,
    const int deviceId,
    std::ostream* ostream /* = &std::cout */) {
  defaultTensorBackend().getMemMgrInfo(msg, deviceId, ostream);
}

void setMemMgrLogStream(std::ostream* stream) {
  defaultTensorBackend().setMemMgrLogStream(stream);
}

void setMemMgrLoggingEnabled(const bool enabled) {
  defaultTensorBackend().setMemMgrLoggingEnabled(enabled);
}

void setMemMgrFlushInterval(const size_t interval) {
  defaultTensorBackend().setMemMgrFlushInterval(interval);
}

} // namespace detail

} // namespace fl
