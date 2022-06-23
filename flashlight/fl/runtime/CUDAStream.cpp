/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/tensor/CUDAUtils.h"

namespace fl {
namespace runtime {

CUDAStream::CUDAStream(const CUDADevice& device, cudaStream_t stream, bool managed) :
  device_(device),
  nativeStream_(stream),
  managed_(managed) {
  // `event_` is used by relativeSync only -- disable timing to reduce overhead
  FL_CUDA_CHECK(cudaEventCreate(&event_, cudaEventDefault | cudaEventDisableTiming));
}

std::shared_ptr<CUDAStream> CUDAStream::create(int flag, bool managed) {
  cudaStream_t nativeStream;
  FL_CUDA_CHECK(cudaStreamCreateWithFlags(&nativeStream, flag));
  auto& manager = DeviceManager::getInstance();
  auto& device = manager.getActiveDevice(DeviceType::CUDA).impl<CUDADevice>();
  auto rawStreamPtr = new CUDAStream(device, nativeStream, managed);
  return std::shared_ptr<CUDAStream>(rawStreamPtr);
}

std::shared_ptr<CUDAStream> CUDAStream::createManaged(int flag) {
  return CUDAStream::create(flag, /* managed */ true);
}

std::shared_ptr<CUDAStream> CUDAStream::createUnmanaged(int flag) {
  return CUDAStream::create(flag, /* managed */ false);
}

std::shared_ptr<CUDAStream> CUDAStream::wrapUnmanaged(
    int deviceId, cudaStream_t stream) {
  auto& manager = DeviceManager::getInstance();
  auto& device =
    manager.getDevice(DeviceType::CUDA, deviceId).impl<CUDADevice>();
  auto rawStreamPtr = new CUDAStream(device, stream, /* managed */ false);
  return std::shared_ptr<CUDAStream>(rawStreamPtr);
}

CUDAStream::~CUDAStream() {
  if (managed_) {
    FL_CUDA_CHECK(cudaStreamDestroy(nativeStream_));
  }
}

const CUDADevice& CUDAStream::device() const {
  return device_;
}

std::future<void> CUDAStream::sync() const {
  return std::async(std::launch::async, [this] {
    FL_CUDA_CHECK(cudaStreamSynchronize(this->nativeStream_));
  });
}

void CUDAStream::relativeSync(const CUDAStream& waitOn) const {
  auto& manager = DeviceManager::getInstance();
  auto* oldActiveCUDADevice = &manager.getActiveDevice(DeviceType::CUDA);
  bool needDeviceSwitch = oldActiveCUDADevice != &device_;
  if (needDeviceSwitch) {
    device_.setActive();
  }
  FL_CUDA_CHECK(cudaEventRecord(event_, waitOn.nativeStream_));
  FL_CUDA_CHECK(cudaStreamWaitEvent(this->nativeStream_, event_, 0));
  if (needDeviceSwitch) {
    oldActiveCUDADevice->setActive();
  }
}

cudaStream_t CUDAStream::handle() const {
  return nativeStream_;
}

} // namespace runtime
} // namespace fl
