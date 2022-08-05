/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/opt/Optimizer.h"

#include <iterator>

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/backend/jit/opt/JitOptimizerExtension.h"
#include "flashlight/fl/tensor/backend/jit/opt/passes/ScalarFolding.h"

namespace fl {

namespace {

template <typename T>
void extend(std::vector<T>& extendee, std::vector<T>&& elems) {
  extendee.insert(
      std::end(extendee),
      std::make_move_iterator(std::begin(elems)),
      std::make_move_iterator(std::end(elems)));
}

} // namespace

Optimizer::Optimizer(TensorBackend& backend) : backend_(backend) {
  // TODO
  // 1. figure out a configuration API (e.g., LLVM pass style macro)
  // 2. think about ordering
  passes_.emplace_back(std::make_unique<ScalarFolding>());
  auto& registrar = detail::TensorExtensionRegistrar::getInstance();
  if (registrar.isTensorExtensionRegistered(
          backend_.backendType(), TensorExtensionType::JitOptimizer)) {
    extend(passes_, backend_.getExtension<JitOptimizerExtension>().passes());
  }
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
