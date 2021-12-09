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
 * Conditionally include autograd extensions
 */
#if FL_USE_CUDNN
#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnAutogradExtension.h"
#endif

namespace fl {

/****************** Autograd Extension Registration ******************/

FL_REGISTER_TENSOR_EXTENSION(
    CudnnAutogradExtension,
    TensorBackendType::ArrayFire);

} // namespace fl
