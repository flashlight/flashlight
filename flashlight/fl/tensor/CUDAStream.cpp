/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/CUDAStream.h"

#include <cuda_runtime.h>

#include "flashlight/fl/tensor/CUDAUtils.h"

namespace fl {

CUDAStream::CUDAStream() : managed_(true) {
  FL_CUDA_CHECK(cudaStreamCreate(&streamHandle_));
}

CUDAStream::CUDAStream(cudaStream_t stream)
    : streamHandle_(stream), managed_(false) {}

CUDAStream::~CUDAStream() {
  // If this obj wasn't created with a CUDA stream, don't destroy it
  if (managed_) {
    FL_CUDA_CHECK(cudaStreamDestroy(streamHandle_));
  }
}

cudaStream_t CUDAStream::handle() const {
  return streamHandle_;
}

std::packaged_task<void()> CUDAStream::sync() const {
  return std::packaged_task<void()>([handle = this->streamHandle_]() -> void {
    FL_CUDA_CHECK(cudaStreamSynchronize(handle));
  });
}

StreamType CUDAStream::type() const {
  return CUDAStream::streamType;
}

} // namespace fl
