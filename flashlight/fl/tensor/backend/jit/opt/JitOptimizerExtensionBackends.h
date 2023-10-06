/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorExtension.h"

#if FL_USE_ONEDNN
  #include "flashlight/fl/tensor/backend/jit/opt/backends/onednn/OneDnnJitOptimizerExtension.h"
#endif // FL_USE_ONEDNN

namespace fl {

/****************** Jit Optimizer Extension Registration ******************/

#if FL_USE_ONEDNN
FL_REGISTER_TENSOR_EXTENSION(OneDnnJitOptimizerExtension, OneDnn);
#endif // FL_USE_ONEDNN

} // namespace fl
