/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <list>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/NodeType.h"
#include "flashlight/fl/tensor/backend/jit/ir/Use.h"

namespace fl {

class ExternalUse;
using ExternalUseList = std::list<ExternalUse*>;
using UseList = std::list<std::unique_ptr<Use>>;
using NodePtr = std::shared_ptr<Node>;

/**
 * A Node represents a step in some Tensor computation.
 *
 * Conceptually, nodes form a computation DAG where the result of the
 * computation is immutable (this must be enforced by all graph rewrites).
 */
class Node {
  std::vector<NodePtr> inputs_;
  std::vector<UseList::iterator> inputUseIters_;
  UseList uses_;
  ExternalUseList externalUses_;
  const Shape shape_;

  // present if this node has been evaluated
  std::optional<Tensor> result_{std::nullopt};

  // set the input at `inputIdx` -- `unlinkInput` should be used to clear the
  // old input at `inputIdx`, if any.
  void linkInput(unsigned inputIdx, NodePtr input);
  void unlinkInput(unsigned inputIdx);

  static void nodeImplTypeCheck(NodeType expect, NodeType actual);

 protected:
  // A constructor that sets up all the metadata
  Node(std::vector<NodePtr>&& inputs, const Shape& shape);

  // Help enforce internal consistency.
  NodePtr getInput(unsigned inputIdx) const;

 public:
  virtual ~Node();

  // Inputs
  const std::vector<NodePtr>& inputs() const;
  void setInput(unsigned inputIdx, NodePtr newInput);

  // Shape
  const Shape& shape() const;

  // Uses
  const UseList& uses() const;
  const ExternalUseList& externalUses() const;
  // replaces both internal and external uses
  void replaceAllUsesWith(NodePtr newInput);

  // Useful for lazy eval
  const std::optional<Tensor>& getResult() const;
  void setResult(Tensor&& tensor);
  void unsetResult();

  // Convenient type checks
  bool isBinary() const;
  bool isCustom() const;
  bool isIndex() const;
  bool isScalar() const;
  bool isValue() const;
  bool isIndexedUpdate() const;

  // Fast & safe casts
  virtual NodeType type() const = 0;

  template <typename T>
  T& impl() {
    nodeImplTypeCheck(T::nodeType, this->type());
    return *static_cast<T*>(this);
  }

  template <typename T>
  const T& impl() const {
    nodeImplTypeCheck(T::nodeType, this->type());
    return *static_cast<const T*>(this);
  }

  /**
   * Cast NodePtr to its actual derived type, useful if caller wishes to obtain
   * a derived NodePtr back.
   *
   * @param node the node to cast from
   * @return node the result of the cast.
   *
   * Example:
   *   NodePtr node = ...;
   *   BinaryNodePtr binaryNode = Node::cast<BinaryNode>(node);
   */
  template <typename T, typename E = typename T::element_type>
  static T cast(NodePtr node) {
    nodeImplTypeCheck(E::nodeType, node->type());
    return std::static_pointer_cast<E>(node);
  }

  friend class ExternalUse;
};

/**
 * A trait for some generic Node functionalities.
 *
 * REQUIRED definition in derived class:
 *   public: static constexpr NodeType nodeType;
 */
template <typename Derived>
class NodeTrait : public Node {
 public:
  NodeTrait(std::vector<NodePtr>&& inputs, const Shape& shape)
      : Node(std::move(inputs), std::move(shape)) {}

  NodeType type() const override {
    return Derived::nodeType;
  }
};

} // namespace fl
