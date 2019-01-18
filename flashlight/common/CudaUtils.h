/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/cuda.h>

namespace fl {
namespace cuda {

/**
 * Gets the Arrayfire CUDA stream. Gets the stream
 * for the device it's called on (with that device id)
 */
cudaStream_t getActiveStream();

} // namespace cuda
} // namespace fl
