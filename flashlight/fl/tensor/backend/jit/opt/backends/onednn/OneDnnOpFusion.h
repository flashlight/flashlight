/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"
#include "flashlight/fl/tensor/backend/jit/opt/Pass.h"

namespace fl {

/**
 * Levearge OneDNN's post-ops to fuse operations.
 *
 * NOTE
 * 1. due to OneDNN limitation, binary post-op only supports rhs argument.
 * 2. currently we avoid recomputation -- fuse iff intermediate nodes are _only_
 *    used as input nodes in the chain. There might be places where benefit of
 *    aggressive fusion outweighs cost of recomputation, need to investigate
 *    more (think Halide).
 *
 * n1   n2
 *  \  /
 *   b2   n3
 *    \  /
 *     b1
 *
 * -->
 *
 *    n1 n2 n3
 *     \ |  /
 * ---------------- CustomNode with evaluation logic that uses OneDNN post-op
 * | n1 = n1 + n2 |
 * | n1 = n2 + n3 |
 * ----------------
 *
 * TODO
 * - leverage commutativity of certain binops to bypass the rhs-only limitation
 *   of OneDNN binary post-ops.
 * - support more than just binop fusion
 */
class OneDnnOpFusion : public Pass {
  struct SearchState;

  // Avoid re-visit, since fuser only need to apply once to each node.
  std::unordered_set<NodePtr> visited_{};

  // 1. Fuse _along_ some path from `node`.
  // 2. recursively optimize other inputs along the fused path.
  NodePtr rewriteFrom(NodePtr node);

  // keep searching for nodes to fuse starting from `node`
  NodePtr searchAndFuse(NodePtr node, SearchState& state);

  // Actual fusion of an op-chain, `node` is a leaf input.
  NodePtr fuseNodes(NodePtr node, SearchState& state);

 public:
  OneDnnOpFusion() = default;
  ~OneDnnOpFusion() = default;

  NodePtr apply(NodePtr root) override;
};

} // namespace fl
