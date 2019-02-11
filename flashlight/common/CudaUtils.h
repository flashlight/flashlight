/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

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

namespace detail {

void check(cudaError_t err, const char* file, int line);

void check(cudaError_t err, const char* prefix, const char* file, int line);

} // namespace detail

} // namespace cuda
} // namespace fl
