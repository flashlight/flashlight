/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <af/cuda.h>

/// usage: `FL_CUDA_CHECK(cudaError_t err[, const char* prefix])`
#define FL_CUDA_CHECK(...) \
  ::fl::cuda::detail::check(__VA_ARGS__, __FILE__, __LINE__)

namespace fl {
namespace cuda {

/**
 * Gets the Arrayfire CUDA stream. Gets the stream
 * for the device it's called on (with that device id)
 */
cudaStream_t getActiveStream();

/**
 * Synchronizes (blocks) a CUDA stream on another. That is, records a snapshot
 * of any operations currently enqueued on CUDA stream blockOn using a CUDA
 * Event, and forces blockee to wait on those events to complete before
 * beginning any future-enqueued operations. Does so without blocking the host
 * CPU thread.
 *
 * @param[in] blockee the CUDA stream to be blocked
 * @param[in] blockOn the CUDA stream to block on, whose events will be waited
 * on to complete before the blockee starts execution of its enqueued events
 * @param[in] event an existing CUDA event to use to record events on blockOn
 * CUDA stream
 */
void synchronizeStreams(
    cudaStream_t blockee,
    cudaStream_t blockOn,
    cudaEvent_t event);

namespace detail {

// Flags for CUDA Event creation. Timing creates overhead, so disable.
constexpr unsigned int kCudaEventDefaultFlags =
    cudaEventDefault | cudaEventDisableTiming;

void check(cudaError_t err, const char* file, int line);

void check(cudaError_t err, const char* prefix, const char* file, int line);

} // namespace detail

} // namespace cuda
} // namespace fl
