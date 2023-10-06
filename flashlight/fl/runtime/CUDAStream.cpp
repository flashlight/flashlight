/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/CUDAStream.h"

#include <cassert>

#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/runtime/DeviceManager.h"

namespace fl {

CUDAStream::CUDAStream(CUDADevice& device, cudaStream_t stream, bool managed)
    : device_(device), nativeStream_(stream), managed_(managed) {
  // Ensure `event_` and `nativeStream_` are associated with the same device
  assert(
      &DeviceManager::getInstance().getActiveDevice(DeviceType::CUDA) ==
      &device);
  // `event_` is used by relativeSync only -- disable timing to reduce overhead
  FL_CUDA_CHECK(
      cudaEventCreate(&event_, cudaEventDefault | cudaEventDisableTiming));
}

std::shared_ptr<CUDAStream> CUDAStream::makeSharedAndRegister(
    CUDADevice& device,
    cudaStream_t stream,
    bool managed) {
  auto rawStreamPtr = new CUDAStream(device, stream, managed);
  auto streamPtr = std::shared_ptr<CUDAStream>(rawStreamPtr);
  device.addStream(streamPtr);
  return streamPtr;
}

std::shared_ptr<CUDAStream> CUDAStream::create(int flag, bool managed) {
  cudaStream_t nativeStream;
  FL_CUDA_CHECK(cudaStreamCreateWithFlags(&nativeStream, flag));
  auto& manager = DeviceManager::getInstance();
  auto& device = manager.getActiveDevice(DeviceType::CUDA).impl<CUDADevice>();
  return makeSharedAndRegister(device, nativeStream, managed);
}

std::shared_ptr<CUDAStream> CUDAStream::createManaged(int flag) {
  return CUDAStream::create(flag, /* managed */ true);
}

std::shared_ptr<CUDAStream> CUDAStream::createUnmanaged(int flag) {
  return CUDAStream::create(flag, /* managed */ false);
}

std::shared_ptr<CUDAStream> CUDAStream::wrapUnmanaged(
    int deviceId,
    cudaStream_t stream) {
  auto& manager = DeviceManager::getInstance();
  const auto& oldActiveDevice = manager.getActiveDevice(DeviceType::CUDA);
  auto& device =
      manager.getDevice(DeviceType::CUDA, deviceId).impl<CUDADevice>();
  // satisfies assumptions of makeSharedAndRegister
  bool needDeviceSwitch = &oldActiveDevice != &device;
  if (needDeviceSwitch) {
    device.setActive();
  }
  auto streamPtr = makeSharedAndRegister(device, stream, /* managed */ false);
  if (needDeviceSwitch) {
    oldActiveDevice.setActive();
  }
  return streamPtr;
}

CUDAStream::~CUDAStream() {
  if (managed_) {
    FL_CUDA_CHECK(cudaStreamDestroy(nativeStream_));
    // Ideally we should unconditionally destroy the event we created, but there
    // is a race hazard between CUDAStream destructor in global context and CUDA
    // shutdown (sometimes the latter may precede the former). So we destroy the
    // event only when it's safe to do so
    FL_CUDA_CHECK(cudaEventDestroy(event_));
  } else {
#ifdef NO_CUDA_STREAM_DESTROY_EVENT
    // Note that this case only results in cuda event "resource leak" if someone
    // creates an unmanaged cuda stream. But managed cuda streams are often used
    // in a global context and released at program shutdown (e.g., for cudnn).
    // So chances of real resource leak is very low.
#else
    FL_CUDA_CHECK(cudaEventDestroy(event_));
#endif
  }
}

const CUDADevice& CUDAStream::device() const {
  return device_;
}

CUDADevice& CUDAStream::device() {
  return device_;
}

void CUDAStream::sync() const {
  FL_CUDA_CHECK(cudaStreamSynchronize(this->nativeStream_));
}

void CUDAStream::relativeSync(const CUDAStream& waitOn) const {
  auto& manager = DeviceManager::getInstance();
  auto* oldActiveCUDADevice = &manager.getActiveDevice(DeviceType::CUDA);
  bool needDeviceSwitch = oldActiveCUDADevice != &device_;
  if (needDeviceSwitch) {
    device_.setActive();
  }
  // event and stream from same instance are guaranteed to have been created
  // from the same device
  FL_CUDA_CHECK(cudaEventRecord(waitOn.event_, waitOn.nativeStream_));
  FL_CUDA_CHECK(cudaStreamWaitEvent(
      this->nativeStream_, waitOn.event_, /* cudaEventWaitDefault = */ 0));
  if (needDeviceSwitch) {
    oldActiveCUDADevice->setActive();
  }
}

cudaStream_t CUDAStream::handle() const {
  return nativeStream_;
}

} // namespace fl
