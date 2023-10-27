/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/opt/backends/onednn/OneDnnOpFusion.h"
#include <optional>

#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnBackend.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"
#include "flashlight/fl/tensor/backend/onednn/Utils.h"

#include "dnnl.hpp"

namespace fl {

struct BinopInfo {
  NodePtr rhsNode;
  BinaryOp op;
};

struct OneDnnOpFusion::SearchState {
  SearchState(NodePtr root, std::vector<BinopInfo> binopInfos)
      : searchRoot(root), accumulatedBinopInfos(binopInfos) {}
  NodePtr searchRoot;
  // Assume `searchRoot == binop2`
  //
  // x0  x1
  //  \  /
  //  binop1  x2
  //     \  /
  //    binop2
  //
  // accumulatedBinopInfos: { { binop2->op(), x2 }, { binop1->op(), x1 } }
  std::vector<BinopInfo> accumulatedBinopInfos;
};

namespace {

// https://github.com/oneapi-src/oneDNN/blob/adeda9fcc20149effb1bffc051262810e9f3138c/include/oneapi/dnnl/dnnl_types.h#L2914
static constexpr unsigned kOneDnnMaxNumPostOps = 32;

dnnl::memory::data_type getOneDnnTypeWithLargestRange(
    const std::vector<const Tensor*>& tensors) {
  dnnl::memory::data_type largestType =
      detail::flToOneDnnType(tensors.front()->type());
  for (unsigned i = 1; i < tensors.size(); i++) {
    auto otherType = detail::flToOneDnnType(tensors[i]->type());
    largestType = detail::getTypeWithLargerRange(largestType, otherType);
  }
  return largestType;
}

std::optional<dnnl::algorithm> tryBinopToOneDnnAlg(const BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
      return dnnl::algorithm::binary_add;
    case BinaryOp::Sub:
      return dnnl::algorithm::binary_sub;
    case BinaryOp::Mul:
      return dnnl::algorithm::binary_mul;
    case BinaryOp::Div:
      return dnnl::algorithm::binary_div;
    case BinaryOp::Eq:
      return dnnl::algorithm::binary_eq;
    case BinaryOp::Neq:
      return dnnl::algorithm::binary_ne;
    case BinaryOp::Gt:
      return dnnl::algorithm::binary_gt;
    case BinaryOp::Gte:
      return dnnl::algorithm::binary_ge;
    case BinaryOp::Lt:
      return dnnl::algorithm::binary_lt;
    case BinaryOp::Lte:
      return dnnl::algorithm::binary_le;
    case BinaryOp::Min:
      return dnnl::algorithm::binary_min;
    case BinaryOp::Max:
      return dnnl::algorithm::binary_max;
    case BinaryOp::Pow:
    case BinaryOp::Mod:
    case BinaryOp::And:
    case BinaryOp::Or:
    case BinaryOp::Shl:
    case BinaryOp::Shr:
    case BinaryOp::BitAnd:
    case BinaryOp::BitOr:
    case BinaryOp::BitXor:
      return std::nullopt;
  }
  throw std::runtime_error(
      "[tryBinopToOneDnnAlg] Unexpected binary operation type");
}

dnnl::algorithm binopToOneDnnAlg(const BinaryOp op) {
  const auto alg = tryBinopToOneDnnAlg(op);
  if (!alg.has_value()) {
    throw std::runtime_error("[binopToOneDnnAlg] unsupported binop for OneDNN");
  }
  return alg.value();
}

bool isOpFusable(const BinaryOp op) {
  return tryBinopToOneDnnAlg(op).has_value();
}

bool isNodeFusable(const NodePtr node) {
  return node->isBinary() && isOpFusable(node->impl<BinaryNode>().op());
}

bool isFusionProfitable(const NodePtr node) {
  // TODO
  // Even if we have multiple uses, it might be possible & profitable to fuse,
  // i.e., recomputation might be okay, think Halide.
  return node->uses().size() + node->externalUses().size() <= 1;
}

bool shouldNodeBeFused(const NodePtr node) {
  return isNodeFusable(node) && isFusionProfitable(node);
}

} // namespace

NodePtr OneDnnOpFusion::rewriteFrom(NodePtr node) {
  SearchState state(node, /* accumulatedBinopInfos = */ {});
  auto fusedNode = searchAndFuse(node, state);
  node->replaceAllUsesWith(fusedNode);
  return fusedNode;
}

NodePtr OneDnnOpFusion::searchAndFuse(NodePtr node, SearchState& state) {
  // TODO for now we just skip shared input, need to think more.
  if (visited_.find(node) != visited_.end() || !shouldNodeBeFused(node) ||
      state.accumulatedBinopInfos.size() > kOneDnnMaxNumPostOps) {
    return fuseNodes(node, state);
  }
  visited_.insert(node);

  if (node->isBinary()) {
    const auto& binaryNode = node->impl<BinaryNode>();
    const auto lhs = binaryNode.lhs();
    const auto rhs = binaryNode.rhs();
    const auto op = binaryNode.op();
    state.accumulatedBinopInfos.push_back({rewriteFrom(rhs), op});
    return searchAndFuse(lhs, state);
  } else {
    // TODO support more fusion for more kinds of op (e.g., reduction)
    throw std::runtime_error(
        "[OneDnnOpFusion::rewriteFrom] If node should be fused, it must be binary node");
  }
}

NodePtr OneDnnOpFusion::fuseNodes(NodePtr node, SearchState& state) {
  for (const auto& input : node->inputs()) {
    rewriteFrom(input);
  }
  // Nothing to fuse, it's one of the following:
  // 1. node
  //
  // 2. node  ...
  //      \  /
  //    searchRoot
  if (state.accumulatedBinopInfos.size() < 2) {
    return state.searchRoot;
  }

  // In the following case `node` is `x1`
  //
  // x1  x2
  //  \  /
  //   op1  x3
  //     \  /
  //     op2
  // becomes
  // inputNodes: { x1, x2, x3 }
  // algs:       { op1, op2 }
  std::vector<NodePtr> inputNodes{node};
  std::vector<dnnl::algorithm> algs;
  for (int i = state.accumulatedBinopInfos.size() - 1; i >= 0; i--) {
    const auto& info = state.accumulatedBinopInfos[i];
    algs.push_back(binopToOneDnnAlg(info.op));
    inputNodes.push_back(info.rhsNode);
  }

  // TODO refactor with common logic in OneDnnBackend
  auto evalFunc = [algs = std::move(algs),
                   dstShape = state.searchRoot->shape()](
                      const std::vector<const Tensor*>& inputs) {
    const Tensor* lhs = inputs[0];
    const Tensor* rhs = inputs[1];
    // NOTE this simulates OneDNNBackend's "typing rule". Once we support type
    // inference in JIT we can get rid of this.
    const auto dstType = getOneDnnTypeWithLargestRange(inputs);

    auto& backend = OneDnnBackend::getInstance();
    auto& engine = backend.engine();

    // prepare memories
    dnnl::algorithm alg = algs.front();
    auto& lhsMem = toOneDnnTensor(*lhs).memory();
    auto& rhsMem = toOneDnnTensor(*rhs).memory();
    const auto lhsMemDesc = lhsMem.get_desc();
    const auto rhsMemDesc = rhsMem.get_desc();
    const auto dstMemDesc =
        detail::oneDnnContiguousMemDescFromShape(dstShape, dstType);
    auto dstMem = dnnl::memory(dstMemDesc, engine);

    // prepare part of arguments
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC_0, lhsMem},
        {DNNL_ARG_SRC_1, rhsMem},
        {DNNL_ARG_DST, dstMem},
    };

    // prepare post ops
    dnnl::post_ops binops;
    for (unsigned i = 1; i < algs.size(); i++) {
      // set up the other input for post-op
      auto& otherMem = toOneDnnTensor(*inputs[i + 1]).memory();
      binops.append_binary(algs[i], otherMem.get_desc());
      args.insert( // DNNL_ARG_SRC_1 feels totally arbitrary...
          {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i - 1) | DNNL_ARG_SRC_1, otherMem});
    }

    // finish building primitive
    dnnl::primitive_attr binaryAttr;
    binaryAttr.set_post_ops(binops);

    // prepare part of primitive
    const dnnl::binary::primitive_desc binaryPrimitiveDesc(
        engine, alg, lhsMemDesc, rhsMemDesc, dstMemDesc, binaryAttr);
    const auto binaryPrimitive = dnnl::binary(binaryPrimitiveDesc);

    // execute primitive
    binaryPrimitive.execute(backend.nativeStream(), args);
    return toTensor<OneDnnTensor>(dstShape, std::move(dstMem));
  };

  return CustomNode::create(
      "OneDnnFusedBinaryOp",
      std::move(inputNodes),
      node->shape(),
      std::move(evalFunc));
}

NodePtr OneDnnOpFusion::apply(NodePtr root) {
  auto optimizedRoot = rewriteFrom(root);
  visited_.clear();
  return optimizedRoot;
}

} // namespace fl
