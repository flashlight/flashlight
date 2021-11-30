/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorExtension.h"

/*
 * Conditionally include vision extensions
 */
#if FL_USE_ARRAYFIRE
#include "flashlight/pkg/vision/tensor/backend/af/VisionExtensionBackend.h"
#endif

namespace fl {

/****************** Vision Extension Registration ******************/

FL_REGISTER_TENSOR_EXTENSION(
    ArrayFireVisionExtension,
    TensorBackendType::ArrayFire);

} // namespace fl
