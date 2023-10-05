/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorExtension.h"

/*
 * Conditionally include vision extensions
 */
#if FL_USE_ARRAYFIRE
  #include "flashlight/pkg/vision/tensor/backend/af/ArrayFireVisionExtension.h"
#endif

namespace fl {

/****************** Vision Extension Registration ******************/

#if FL_USE_ARRAYFIRE
FL_REGISTER_TENSOR_EXTENSION(ArrayFireVisionExtension, ArrayFire);
#endif // FL_USE_ARRAYFIRE

} // namespace fl
