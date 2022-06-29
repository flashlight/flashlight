/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/CUDAUtils.h"

#include <sstream>
#include <stdexcept>

#include "flashlight/fl/runtime/CUDAStream.h"
// TODO: remove me after removing the dependency on Tensor
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {
namespace cuda {

// TODO{fl::Tensor}{CUDA} remove the dependency on Tensor so this can be
// moved to a runtime abstraction
cudaStream_t getActiveStream() {
  return Tensor().stream().impl<runtime::CUDAStream>().handle();
}

void synchronizeStreams(
    cudaStream_t blockee,
    cudaStream_t blockOn,
    cudaEvent_t event) {
  FL_CUDA_CHECK(cudaEventRecord(event, blockOn));
  FL_CUDA_CHECK(cudaStreamWaitEvent(blockee, event, 0));
}

namespace detail {

void check(cudaError_t err, const char* file, int line) {
  check(err, "", file, line);
}

void check(cudaError_t err, const char* prefix, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream ess;
    ess << prefix << '[' << file << ':' << line
        << "] CUDA error: " << cudaGetErrorString(err);
    throw std::runtime_error(ess.str());
  }
}

} // namespace detail

} // namespace cuda
} // namespace fl
