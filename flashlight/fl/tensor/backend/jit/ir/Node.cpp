/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

#include "flashlight/fl/tensor/backend/jit/ir/ExternalUse.h"

#include <cassert>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace fl {

void Node::nodeImplTypeCheck(NodeType expect, NodeType actual) {
  if (expect != actual) {
    std::ostringstream oss;
    oss << "[fl::Node::nodeImplTypeCheck] "
        << "specified node type: [" << actual << "] "
        << "doesn't match actual node type: [" << expect << "]";
    throw std::invalid_argument(oss.str());
  }
}

void Node::linkInput(unsigned inputIdx, NodePtr input) {
  inputs_.at(inputIdx) = input;
  // update metadata
  auto inputUseIter = input->uses_.emplace(
      input->uses_.end(), std::make_unique<Use>(*this, inputIdx));
  inputUseIters_[inputIdx] = inputUseIter;
}

void Node::unlinkInput(unsigned inputIdx) {
  const auto oldInput = inputs_.at(inputIdx);
  const auto oldInputUseIter = inputUseIters_[inputIdx];
  assert(oldInput && "No input at given index");
  // reset
  inputUseIters_[inputIdx] = oldInput->uses_.end();
  inputs_[inputIdx] = nullptr;
  // clean up metadata
  oldInput->uses_.erase(oldInputUseIter);
}

Node::Node(std::vector<NodePtr>&& inputs, const Shape& shape)
    : inputs_(inputs), shape_(shape) {
  inputs_.resize(inputs.size());
  inputUseIters_.resize(inputs.size());
  for (unsigned inputIdx = 0; inputIdx < inputs.size(); inputIdx++) {
    linkInput(inputIdx, inputs[inputIdx]);
  }
}

Node::~Node() {
  for (unsigned inputIdx = 0; inputIdx < inputs_.size(); inputIdx++) {
    unlinkInput(inputIdx);
  }
}

NodePtr Node::getInput(unsigned inputIdx) const {
  return inputs_.at(inputIdx);
}

const std::vector<NodePtr>& Node::inputs() const {
  return inputs_;
}

void Node::setInput(unsigned inputIdx, NodePtr newInput) {
  unlinkInput(inputIdx);
  linkInput(inputIdx, newInput);
}

const Shape& Node::shape() const {
  return shape_;
}

const UseList& Node::uses() const {
  return uses_;
}

const ExternalUseList& Node::externalUses() const {
  return externalUses_;
}

void Node::replaceAllUsesWith(NodePtr newNode) {
  if (newNode.get() != this) {
    for (const auto& use : uses_) {
      auto& userNode = use->user();
      userNode.inputs_[use->inputIdx()] = newNode;
    }
    newNode->uses_.splice(newNode->uses_.begin(), this->uses_);
    for (const auto& externalUse : externalUses_) {
      externalUse->usee_ = newNode;
    }
    newNode->externalUses_.splice(
        newNode->externalUses_.begin(), this->externalUses_);
  }
}

const std::optional<Tensor>& Node::getResult() const {
  return result_;
}

void Node::setResult(Tensor&& tensor) {
  if (result_.has_value()) {
    throw std::invalid_argument("[Node::setResult] Result already set");
  } else {
    result_ = std::move(tensor);
  }
}

void Node::unsetResult() {
  if (result_.has_value()) {
    result_ = std::nullopt;
  } else {
    throw std::invalid_argument("[Node::unsetResult] Result not set");
  }
}

bool Node::isBinary() const {
  return type() == NodeType::Binary;
}

bool Node::isCustom() const {
  return type() == NodeType::Custom;
}

bool Node::isIndex() const {
  return type() == NodeType::Index;
}

bool Node::isIndexedUpdate() const {
  return type() == NodeType::IndexedUpdate;
}

bool Node::isScalar() const {
  return type() == NodeType::Scalar;
}

bool Node::isValue() const {
  return type() == NodeType::Value;
}

} // namespace fl
