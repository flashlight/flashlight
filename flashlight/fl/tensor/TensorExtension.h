/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
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
 * A runtime type denoting the tensor extension.
 */
enum class TensorExtensionType {
  Generic, // placeholder
  Autograd,
  Vision,
  JitOptimizer,
};

// Common base type
class TensorExtensionBase {
 public:
  virtual ~TensorExtensionBase() = default;

  virtual bool isDataTypeSupported(const fl::dtype& dtype) const = 0;
};

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

  // TODO(jacobkahn): change this to an array and have indices for extension
  // types correspond to extension instances
  std::unordered_map<
      TensorBackendType,
      std::unordered_map<TensorExtensionType, TensorExtensionCallback>>
      extensions_;

 public:
  bool registerTensorExtension(
      TensorBackendType backend,
      TensorExtensionType extensionType,
      TensorExtensionCallback&& creationFunc);

  static TensorExtensionRegistrar& getInstance();
  ~TensorExtensionRegistrar() = default;

  template <typename T>
  bool registerTensorExtension(TensorBackendType backend) {
    // TODO: use a static T::create instead of a lambda if we can enforce its
    // declaration and definition on interface functions
    return this->registerTensorExtension(
        backend, T::getExtensionType(), []() -> std::unique_ptr<T> {
          return std::make_unique<T>();
        });
  }

  bool isTensorExtensionRegistered(
      TensorBackendType backend,
      TensorExtensionType extensionType);

  TensorExtensionCallback& getTensorExtensionCreationFunc(
      TensorBackendType backend,
      TensorExtensionType extensionType);
};

} // namespace detail

/**
 * Register a tensor extension. Template type T is the type of the tensor
 * extension
 *
 * @param[in] backendType the type of the backend to register the extension to.
 * See TensorBackendType.
 */
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

template <typename T>
struct TensorExtensionRegisterer {
  TensorExtensionRegisterer(TensorBackendType t) {
    ::fl::registerTensorExtension<T>(t);
  }
};

/**
 * Register a tensor extension.
 *
 * @param[in] T the class type of the tensor extension
 * @param[in] backendType the type of the backend to register the extension to.
 * See TensorBackendType.
 */
#define FL_REGISTER_TENSOR_EXTENSION(T, BACKEND_TYPE) \
  TensorExtensionRegisterer<T> T##BACKEND_TYPE(TensorBackendType::BACKEND_TYPE)

} // namespace fl
