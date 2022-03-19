/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Stream.h"

#include <cuda_runtime.h>

namespace fl {

/**
 * A small abstraction around a CUDA stream.
 */
class CUDAStream : public StreamImpl {
  cudaStream_t streamHandle_;
  const bool managed_{false}; // destroy stream on destruction

 public:
  static constexpr StreamType streamType = StreamType::CUDA;

  /**
   * Construct a stream.
   */
  CUDAStream();

  /**
   * Construct a CUDAStream from an existing handle. If this constructor is
   * used, the stream won't be destroyed when this object is destroyed.
   *
   * @param[in] stream the CUDA stream with which to create this stream.
   */
  explicit CUDAStream(cudaStream_t stream);

  ~CUDAStream() override;

  /**
   * Get the underlying cudaStream_t handle.
   */
  cudaStream_t handle() const;

  /**
   * Return a future which will perform blocking synchronization (equivalent to
   * cudaStreamSynchronize) if waited on.
   */
  std::packaged_task<void()> sync() const override;

  StreamType type() const override;
};

} // namespace fl
