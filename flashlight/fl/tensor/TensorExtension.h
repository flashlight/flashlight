/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <type_traits>
#include <unordered_map>

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/**
 * Document me
 */
enum class TensorExtensionType {
  Generic = 0,
  Vision = 1,
};

// Common base type
class TensorExtensionBase {};

namespace detail {

using TensorExtensionCallback =
    std::function<std::unique_ptr<TensorExtensionBase>()>;

/**
 * Employ an extensible factory singleton pattern to handle creation callbacks
 * for creating specific TensorExtension instances.
 *
 * Users should not directly use this singleton and should instead
 */
class TensorExtensionRegistrar {
  // Intentionally private. Only one instance should exist/it should be accessed
  // via getInstance().
  TensorExtensionRegistrar() = default;

  std::unordered_map<
      TensorBackendType,
      std::unordered_map<TensorExtensionType, TensorExtensionCallback>>
      extensions_;

  /*
   * Document me
   */
  bool registerTensorExtension(
      TensorBackendType backend,
      TensorExtensionType extensionType,
      TensorExtensionCallback&& creationFunc);

 public:
  static TensorExtensionRegistrar& getInstance();
  ~TensorExtensionRegistrar() = default;

  /*
   * Document me
   */
  template <typename T>
  bool registerTensorExtension(TensorBackendType backend) {
    // TODO: use a static T::create instead of a lambda if we can enforce its
    // declaration and definition on interface functions
    return this->registerTensorExtension(
        backend, T::getExtensionType(), []() -> std::unique_ptr<T> {
          return std::make_unique<T>();
        });
  }

  /*
   * Document me
   */
  TensorExtensionCallback& getTensorExtensionCreationFunc(
      TensorBackendType backend,
      TensorExtensionType extensionType);
};

} // namespace detail

template <typename T>
bool registerTensorExtension(TensorBackendType backendType) {
  return detail::TensorExtensionRegistrar::getInstance()
      .registerTensorExtension<T>(backendType);
}

template <typename T>
class TensorExtension : public TensorExtensionBase {
 public:
  static TensorExtensionType getExtensionType() {
    return T::extensionType;
  }
};

} // namespace fl
