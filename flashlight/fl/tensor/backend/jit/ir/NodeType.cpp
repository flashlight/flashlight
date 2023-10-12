/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/NodeType.h"

#include <stdexcept>

namespace fl {

std::string nodeTypeToString(const NodeType type) {
  switch (type) {
    case NodeType::Binary:
      return "Binary";
    case NodeType::Custom:
      return "Custom";
    case NodeType::Scalar:
      return "Scalar";
    case NodeType::Value:
      return "Value";
    case NodeType::Index:
      return "Index";
    case NodeType::IndexedUpdate:
      return "IndexedUpdate";
  }
  throw std::runtime_error("Unknown node type");
}

std::ostream& operator<<(std::ostream& os, const NodeType& type) {
  return os << nodeTypeToString(type);
}

} // namespace fl
