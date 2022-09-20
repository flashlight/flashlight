/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if FL_USE_ARRAYFIRE
  #include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"
  #include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#endif
#if FL_USE_TENSOR_STUB
  #include "flashlight/fl/tensor/backend/stub/StubBackend.h"
  #include "flashlight/fl/tensor/backend/stub/StubTensor.h"
#endif

namespace fl {

#if FL_USE_ARRAYFIRE
/**
 * The default tensor type in Flashlight. Currently ArrayFire.
 */
using DefaultTensorType_t = fl::ArrayFireTensor;
using DefaultTensorBackend_t = fl::ArrayFireBackend;
#else
  #if FL_USE_TENSOR_STUB
using DefaultTensorType_t = fl::StubTensor;
using DefaultTensorBackend_t = fl::StubBackend;
  #endif
#endif

/**
 * Returns a TensorBackend instance for the default tensor type, even if changed
 * at runtime.
 */
TensorBackend& defaultTensorBackend();

} // namespace fl
