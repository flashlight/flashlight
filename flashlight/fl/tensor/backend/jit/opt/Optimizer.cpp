/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/opt/Optimizer.h"

#include "flashlight/fl/tensor/backend/jit/opt/passes/ScalarFolding.h"

namespace fl {

Optimizer::Optimizer() {
  // TODO figure out a configuration API (e.g., LLVM pass style macro)
  passes_.emplace_back(std::make_unique<ScalarFolding>());
}

Node* Optimizer::optimize(Node* node) {
  // TODO use an `ExternalUse` interface to enable `Node::replaceAllUsesWith()`
  // to update JitTensorBase::node() as well. We don't want to store these
  // "external" uses together with node uses in `Node::uses()` because the
  // latter is meant to help reason about JIT graph structure.
  //
  // Once we do that, `Optimizer::optimize` and `Pass::apply` don't have to
  // return a Node* anymore, and relevant refcount management gets cleaner too.
  Node* currNode = node;
  bool currNodeMustBeDeleted = false;
  for (const auto& pass : passes_) {
    Node* nextNode = pass->apply(currNode);
    // intermediate nodes must be deleted -- caller only gets final output node
    if (currNode != node && currNode != nextNode && currNodeMustBeDeleted) {
      delete currNode;
    }
    currNode = nextNode;
    currNodeMustBeDeleted = currNode->getRefCount() == 0;
  }
  return currNode;
}

} // namespace fl
