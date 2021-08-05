/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorAdapter.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"

#if FL_USE_ARRAYFIRE
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#endif

/**
 * The default tensor type in Flashlight. Currently ArrayFire.
 */
using DefaultTensorType_t = fl::ArrayFireTensor;

/**
 * The compile time value which will be true if the default backend is
 * available.
 */
#define FL_DEFAULT_BACKEND_COMPILE_FLAG FL_USE_ARRAYFIRE

namespace fl {
namespace detail {

DefaultTensorType& DefaultTensorType::getInstance() {
  static DefaultTensorType instance;
  return instance;
}

DefaultTensorType::DefaultTensorType() {
  creationFunc_ = [](const Shape& shape,
                     fl::dtype type,
                     void* ptr,
                     MemoryLocation memoryLocation) {
  // Resolve the default backend in order of preference/availability
#if FL_DEFAULT_BACKEND_COMPILE_FLAG
    return std::make_unique<DefaultTensorType_t>(
        shape, type, ptr, memoryLocation);
#else
    throw std::runtime_error(
        "Cannot construct tensor: Flashlight built "
        "without an available tensor backend.");

#endif
  };
}

void DefaultTensorType::setCreationFunc(DefaultTensorTypeFunc_t&& func) {
  creationFunc_ = std::move(func);
}

const DefaultTensorTypeFunc_t& DefaultTensorType::getCreationFunc() const {
  return creationFunc_;
}

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
  return DefaultTensorType::getInstance().getCreationFunc()(
      shape, type, ptr, memoryLocation);
}

} // namespace detail
} // namespace fl
