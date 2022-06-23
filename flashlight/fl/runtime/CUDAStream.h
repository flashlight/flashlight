/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/runtime/CUDADevice.h"
#include "flashlight/fl/runtime/Stream.h"

#include <cuda_runtime.h>

namespace fl {
namespace runtime {

/**
 * An abstraction for CUDA stream with controlled creation methods.
 */
class CUDAStream : public StreamTrait<CUDAStream> {
  // the device upon which the underlying native stream was created
  const CUDADevice& device_;
  // the underlying native stream
  cudaStream_t nativeStream_;
  // whether the native stream's lifetime is managed by this object
  const bool managed_;
  // re-used for relative synchronization to reduce overhead
  cudaEvent_t event_;

  // A barebone constructor which just initializes the fields.
  CUDAStream(const CUDADevice& device, cudaStream_t stream, bool managed = false);

  // A fully configurable create, hidden for internal use.
  static std::shared_ptr<CUDAStream> create(int flag, bool managed);

 public:
  // prevent name hiding
  using StreamTrait<CUDAStream>::relativeSync;

  static constexpr StreamType type = StreamType::CUDA;

  /**
   * Creates an unmanaged wrapper around an existing native CUDA stream.
   *
   * @param[in] deviceId the native device ID upon which `stream` was created.
   * @param[in] stream the underlying CUDA stream.
   * @param[in] managed whether the lifetime of the created native stream will
   * be managed by this object.
   *
   * @return a shared pointer to a CUDAStream that wraps around the given native
   * stream.
   */
  static std::shared_ptr<CUDAStream> wrapUnmanaged(
      int deviceId, cudaStream_t stream);

    /**
     * Create a managed CUDAStream around an internally created native CUDA
     * stream.
     *
     * @param[in] flag the flag used for creating native CUDA stream.
     *
     * @return a shared pointer to the CUDAStream created.
     */
  static std::shared_ptr<CUDAStream> createManaged(
      int flag = cudaStreamDefault);

    /**
     * Create an unmanaged CUDAStream around an internally created native CUDA
     * stream.
     *
     * @param[in] flag the flag used for creating native CUDA stream.
     *
     * @return a shared pointer to the CUDAStream created.
     */
  static std::shared_ptr<CUDAStream> createUnmanaged(
      int flag = cudaStreamDefault);

  /**
   * Destroy any stream managed by this object.
   */
  ~CUDAStream() override;

  const CUDADevice& device() const override;
  std::future<void> sync() const override;
  void relativeSync(const CUDAStream& waitOn) const override;

  /**
   * Get the native CUDA stream handle.
   *
   * @return the native CUDA stream handle.
   */
  cudaStream_t handle() const;
};

} // namespace runtime
} // namespace fl
