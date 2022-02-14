/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Compute.h"

#include <utility>

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

void sync() {
  Tensor().backend().sync();
}

void sync(const int deviceId) {
  Tensor().backend().sync(deviceId);
}

void eval(Tensor& tensor) {
  Tensor().backend().eval(tensor);
}

int getDevice() {
  return Tensor().backend().getDevice();
}

void setDevice(const int deviceId) {
  Tensor().backend().setDevice(deviceId);
}

int getDeviceCount() {
  return Tensor().backend().getDeviceCount();
}

namespace detail {

void getMemMgrInfo(
    const char* msg,
    const int deviceId,
    std::ostream* ostream /* = &std::cout */) {
  Tensor().backend().getMemMgrInfo(msg, deviceId, ostream);
}

void setMemMgrLogStream(std::ostream* stream) {
  Tensor().backend().setMemMgrLogStream(stream);
}

void setMemMgrLoggingEnabled(const bool enabled) {
  Tensor().backend().setMemMgrLoggingEnabled(enabled);
}

void setMemMgrFlushInterval(const size_t interval) {
  Tensor().backend().setMemMgrFlushInterval(interval);
}

} // namespace detail

} // namespace fl
