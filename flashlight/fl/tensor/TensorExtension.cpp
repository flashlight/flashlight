/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorExtension.h"

namespace fl {
namespace detail {

bool TensorExtensionRegistrar::registerTensorExtension(
    TensorBackendType backend,
    TensorExtensionType extensionType,
    TensorExtensionCallback&& creationFunc) {
  if (extensions_.find(backend) == extensions_.end()) {
    extensions_.emplace(
        backend,
        std::unordered_map<TensorExtensionType, TensorExtensionCallback>());
  }
  auto& _extensions = extensions_[backend];

  // noop if already exists
  if (_extensions.find(extensionType) != _extensions.end()) {
    return true;
  }

  // Add extension to registry
  _extensions.emplace(extensionType, std::move(creationFunc));
  return true;
}

TensorExtensionCallback&
TensorExtensionRegistrar::getTensorExtensionCreationFunc(
    TensorBackendType backend,
    TensorExtensionType extensionType) {
  if (extensions_.find(backend) == extensions_.end()) {
    throw std::invalid_argument(
        "TensorExtensionRegistrar::getTensorExtensionCreationFunc: "
        "no tensor extensions registered for given backend.");
  }
  auto& _extensions = extensions_[backend];
  if (_extensions.find(extensionType) == _extensions.end()) {
    throw std::invalid_argument(
        "TensorExtensionRegistrar::getTensorExtensionCreationFunc: "
        "given extension type is not registered for this backend.");
  }
  return _extensions[extensionType];
}

TensorExtensionRegistrar& TensorExtensionRegistrar::getInstance() {
  static TensorExtensionRegistrar instance;
  return instance;
}

} // namespace detail
} // namespace fl
