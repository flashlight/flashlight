/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

#include <variant>

namespace fl {

/**
 * A node that represents scalar of specific shape & type.
 */
class ScalarNode : public NodeTrait<ScalarNode> {
  // these types can hold all types scalars FL support, w/o loss of precision
  using ScalarType = std::variant<long long, double, unsigned long long>;

  const Shape shape_;
  const dtype dtype_;
  const ScalarType scalar_; // value used for initialization

  // intentionally kept private to control allocation
  ScalarNode(const Shape& shape, const dtype type, const ScalarType scalar);

 public:
  static constexpr NodeType nodeType = NodeType::Scalar;

  template <typename T>
  static ScalarNode*
  create(const Shape& shape, const dtype type, const T scalar) {
    switch (type) {
      case dtype::b8:
      case dtype::s16:
      case dtype::s32:
      case dtype::s64:
      case dtype::u8:
      case dtype::u16:
      case dtype::u32:
        return new ScalarNode(shape, type, static_cast<long long>(scalar));
      case dtype::u64:
        return new ScalarNode(
            shape, type, static_cast<unsigned long long>(scalar));
      case dtype::f16:
      case dtype::f32:
      case dtype::f64:
        return new ScalarNode(shape, type, static_cast<double>(scalar));
    }
    throw std::runtime_error("[ScalarNode::create] Unknown dtype");
  }

  const Shape& shape() const;
  dtype dataType() const;

  // cast to T
  template <typename T>
  T scalar() const {
    return std::visit([](auto&& val) { return static_cast<T>(val); }, scalar_);
  }
};

} // namespace fl
