/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorAdapter.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"


namespace fl::detail {

DefaultTensorType& DefaultTensorType::getInstance() {
  static DefaultTensorType instance;
  return instance;
}

DefaultTensorType::DefaultTensorType() {
  // Resolve the default backend in order of preference/availability
  // See DefaultTensorType.h
#if FL_DEFAULT_BACKEND_COMPILE_FLAG
  creationFunc_ = std::make_unique<TensorCreatorImpl<DefaultTensorType_t>>();
#else
  throw std::runtime_error(
      "Cannot construct DefaultTensorType singleton: Flashlight built "
      "without an available tensor backend.");

#endif
}

std::unique_ptr<TensorCreator> DefaultTensorType::swap(
    std::unique_ptr<TensorCreator> creator) noexcept {
  std::unique_ptr<TensorCreator> old = std::move(creationFunc_);
  creationFunc_ = std::move(creator);
  return old;
}

const TensorCreator& DefaultTensorType::getTensorCreator() const {
  return *creationFunc_;
}

} // namespace fl
