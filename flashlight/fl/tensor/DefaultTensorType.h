/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if FL_USE_ARRAYFIRE
  #include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"
  #include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#endif
#if FL_USE_ONEDNN
  #include "flashlight/fl/tensor/backend/onednn/OneDnnBackend.h"
  #include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"
#endif
#if FL_USE_TENSOR_STUB
  #include "flashlight/fl/tensor/backend/stub/StubBackend.h"
  #include "flashlight/fl/tensor/backend/stub/StubTensor.h"
#endif

namespace fl {

/**
 * Select the default tensor type in Flashlight. Currently ArrayFire.
 *
 * FL_DEFAULT_BACKEND_COMPILE_FLAG is the compile time value which will
 * be true if the default backend is available.
 */
#if FL_USE_ARRAYFIRE
using DefaultTensorType_t = fl::ArrayFireTensor;
using DefaultTensorBackend_t = fl::ArrayFireBackend;
#define FL_DEFAULT_BACKEND_COMPILE_FLAG FL_USE_ARRAYFIRE

#elif FL_USE_ONEDNN
using DefaultTensorType_t = fl::OneDnnTensor;
using DefaultTensorBackend_t = fl::OneDnnBackend;
#define FL_DEFAULT_BACKEND_COMPILE_FLAG FL_USE_ONEDNN

#elif FL_USE_TENSOR_STUB
using DefaultTensorType_t = fl::StubTensor;
using DefaultTensorBackend_t = fl::StubBackend;
#define FL_DEFAULT_BACKEND_COMPILE_FLAG FL_USE_TENSOR_STUB

#else
#error Unreachable - no tensor backend selected.
#endif

/**
 * Returns a TensorBackend instance for the default tensor type, even if changed
 * at runtime.
 */
TensorBackend& defaultTensorBackend();

} // namespace fl
