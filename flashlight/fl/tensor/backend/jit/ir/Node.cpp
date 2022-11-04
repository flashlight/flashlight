/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace fl {

void Node::nodeImplTypeCheck(NodeType expect, NodeType actual) const {
  if (expect != actual) {
    std::ostringstream oss;
    oss << "[fl::Node::impl] "
        << "specified node type: [" << actual << "] "
        << "doesn't match actual node type: [" << expect << "]";
    throw std::invalid_argument(oss.str());
  }
}

void Node::setInputImpl(unsigned inputIdx, Node* input) {
  inputs_.at(inputIdx) = input;
  // update metadata
  auto inputUse = Use::create(this, inputIdx);
  auto inputUseIter = input->uses_.insert(input->uses_.end(), inputUse);
  inputUseIters_[inputIdx] = inputUseIter;
  input->incRefCount();
}

void Node::resetInput(unsigned inputIdx) {
  const auto oldInput = inputs_.at(inputIdx);
  const auto oldInputUseIter = inputUseIters_[inputIdx];
  assert(oldInput && "No input at given index");
  // reset
  inputUseIters_[inputIdx] = oldInput->uses_.end();
  inputs_[inputIdx] = nullptr;
  // clean up metadata
  delete *oldInputUseIter;
  oldInput->uses_.erase(oldInputUseIter);
  oldInput->decRefCount();
}

Node::Node(std::vector<Node*>&& inputs) {
  inputs_.resize(inputs.size());
  inputUseIters_.resize(inputs.size());
  for (unsigned inputIdx = 0; inputIdx < inputs.size(); inputIdx++) {
    setInputImpl(inputIdx, inputs[inputIdx]);
  }
}

Node::~Node() {
  for (unsigned inputIdx = 0; inputIdx < inputs_.size(); inputIdx++) {
    resetInput(inputIdx);
  }
}

Node* Node::getInput(unsigned inputIdx) const {
  return inputs_.at(inputIdx);
}

const std::vector<Node*>& Node::inputs() const {
  return inputs_;
}

void Node::setInput(unsigned inputIdx, Node* newInput) {
  resetInput(inputIdx);
  setInputImpl(inputIdx, newInput);
}

const UseList& Node::uses() const {
  return uses_;
}

void Node::replaceAllUsesWith(Node* newInput) {
  if (newInput != this) {
    // each iteration updates links an existing user to newInput
    while (!uses_.empty()) {
      const auto* nextUse = *uses_.begin();
      nextUse->user()->setInput(nextUse->inputIdx(), newInput);
    }
  }
}

unsigned Node::getRefCount() const {
  return refCount_;
}

void Node::incRefCount() {
  refCount_++;
}

void Node::decRefCount() {
  if (refCount_ == 0) {
    throw std::runtime_error("[Node::decRefCount] Refcount already 0");
  }
  refCount_--;
  if (refCount_ == 0) {
    delete this;
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

bool Node::isScalar() const {
  return type() == NodeType::Scalar;
}

bool Node::isValue() const {
  return type() == NodeType::Value;
}

} // namespace fl
