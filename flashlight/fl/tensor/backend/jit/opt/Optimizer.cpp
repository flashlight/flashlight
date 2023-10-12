/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/opt/Optimizer.h"

#include <iterator>

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/backend/jit/opt/JitOptimizerExtension.h"
#include "flashlight/fl/tensor/backend/jit/opt/JitOptimizerExtensionBackends.h"
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

NodePtr Optimizer::optimize(NodePtr node) {
  for (const auto& pass : passes_) {
    node = pass->apply(node);
  }
  return node;
}

} // namespace fl
