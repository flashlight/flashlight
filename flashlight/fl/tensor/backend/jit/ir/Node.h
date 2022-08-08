/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <list>
#include <optional>
#include <vector>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/NodeType.h"
#include "flashlight/fl/tensor/backend/jit/ir/Use.h"

namespace fl {

using UseList = std::list<Use*>;

/**
 * A Node represents a step in some Tensor computation.
 *
 * Conceptually, nodes form a computation DAG where the result of the
 * computation is immutable (this must be enforced by all graph rewrites).
 *
 * Ownership model:
 * 1. Node has shared ownership via manual refcount update.
 * 2. Use is owned by user node.
 *
 * Possible stages of lifetime
 * 1. After the initial node creation (subclass must enforce heap allocation),
 *    the creator is the owner (responsible for manually deleting the node).
 * 2. The first `incRefCount` call promotes the node into managed lifetime, and
 *    caller becomes the first owner in the new shared ownership.
 * 3. Ensuing `(inc/dec)RefCount` calls takes/releases the shared ownership
 * 4. The last `decRefCount` call will delete the node.
 */
class Node {
  std::vector<Node*> inputs_;
  std::vector<UseList::iterator> inputUseIters_;
  UseList uses_;
  const Shape shape_;
  unsigned refCount_{0};

  // present if this node has been evaluated
  std::optional<Tensor> result_{std::nullopt};

  void nodeImplTypeCheck(NodeType expect, NodeType actual) const;

  // set the input at `inputIdx` -- `resetInput` should be used to clear the old
  // input at `inputIdx`, if any.
  void setInputImpl(unsigned inputIdx, Node* input);

  // "unlink" the input at `inputIdx`
  void resetInput(unsigned inputIdx);

 protected:
  // A constructor that sets up all the metadata
  Node(std::vector<Node*>&& inputs, const Shape& shape);

  // Help enforce internal consistency.
  Node* getInput(unsigned inputIdx) const;

 public:
  virtual ~Node();

  // Inputs
  const std::vector<Node*>& inputs() const;
  void setInput(unsigned inputIdx, Node* newInput);

  // Shape
  const Shape& shape() const;

  // Uses
  const UseList& uses() const;
  void replaceAllUsesWith(Node* newInput);

  // Mainly for debugging/testing
  unsigned getRefCount() const;
  // Use carefully -- this manually simulates shared ownership of a node
  void incRefCount();
  void decRefCount();

  // Useful for lazy eval
  const std::optional<Tensor>& getResult() const;
  void setResult(Tensor&& tensor);
  void unsetResult();

  // Convenient type checks
  bool isBinary() const;
  bool isCustom() const;
  bool isScalar() const;
  bool isValue() const;

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
  NodeTrait(std::vector<Node*>&& inputs, const Shape& shape)
    : Node(std::move(inputs), std::move(shape)) {}

  NodeType type() const override {
    return Derived::nodeType;
  }
};

} // namespace fl
