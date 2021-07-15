/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorAdapter.h"

#include <memory>
#include <stdexcept>

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"

#if FL_USE_ARRAYFIRE
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#endif

/**
 * The default tensor backend in Flashlight. Currently ArrayFire.
 */
using DefaultBackend = fl::ArrayFireTensor;

/**
 * The compile time value which will be true if the default backend is
 * available.
 */
#define FL_DEFAULT_BACKEND_COMPILE_FLAG FL_USE_ARRAYFIRE

namespace fl {
namespace detail {

/*
 * Resolve the default tensor backend based on compile-time dependencies.
 *
 * For now, ArrayFire is required. If not available, throw.
 */
std::unique_ptr<TensorAdapterBase> getDefaultAdapter(
    const Shape& shape /* = Shape() */,
    fl::dtype type /* = fl::dtype::f32 */,
    void* ptr /* = nullptr */,
    MemoryLocation memoryLocation /* = Location::Host */) {
#if FL_DEFAULT_BACKEND_COMPILE_FLAG
  return std::make_unique<DefaultBackend>(shape, type, ptr, memoryLocation);
#else
  throw std::runtime_error(
      "Cannot construct tensor: Flashlight built "
      "without an available tensor backend.");
#endif
}

} // namespace detail
} // namespace fl
