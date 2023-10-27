/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"

#include <memory>
#include <sstream>
#include <stdexcept>

#include "flashlight/fl/tensor/backend/jit/ir/ExternalUse.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexedUpdateNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

#define FL_JIT_TENSOR_UNIMPLEMENTED \
  throw std::invalid_argument(      \
      "JitTensorBase::" + std::string(__func__) + " - unimplemented.");

namespace fl {

// represents the data referred to by indexinges
class DataStorage {
  ExternalUse externalUse_;

 public:
  DataStorage(NodePtr node) : externalUse_(node) {}

  void replaceNode(NodePtr newNode) {
    externalUse_.setUsee(newNode);
  }

  NodePtr node() {
    return externalUse_.usee();
  }
};

// data shared among shallow copies and views of a tensor.
class JitTensorBase::SharedData {
  // 1. Why `shared_ptr<DataStorage> dataStorage_`?
  //   Indexing essentially creates a shallow-copy that views the indexed region
  //   of the original tensor. So any update to the original tensor should
  //   reflect as an update to dataNode for all shallow copies.
  // 2. Why `indexings_`?
  //   Underlying data node update requires creating a __new__ viewNode_ based
  //   on the __old__ indexings and __new__ data node (recall SSA semantics).
  // 3. Why `oldDataNode_`?
  //   When underlying data node changes, not all SharedData objects are
  //   informed directly, so we lazily perform (2), and use `oldDataNode_` to
  //   guide us.
  std::shared_ptr<DataStorage> dataStorage_;
  // TODO consider
  // 1. using an immutable linked-list here to speed things up
  // 2. making `std::vector<Index>` into an immutable class and use it as
  //    shared_ptr (to avoid copying when passing to ViewNodoe), since it's all
  //    readonly -- JitTensor gives us free immutability for Tensor index. To
  //    push it further, we can use custom shared data structure that has a more
  //    efficient copy constructor/uses std::atomic with
  //    std::memory_order::memory_order_relaxed on a refcount to reduce
  //    contention overhead.
  // 3. index merging, e.g., (1, 3)(1, 2) --> (2, 3). Maybe as an optimization
  //    pass, need to think more.
  std::vector<std::vector<Index>> indexings_{};
  std::optional<NodePtr> oldDataNode_{std::nullopt}; // None iff no indexings
  std::optional<NodePtr> viewNode_{std::nullopt}; // None iff no indexings

  void updateViewNodeIfNeeded() {
    if (indexings_.empty() || dataStorage_->node() == oldDataNode_) {
      return;
    }
    // apply index one by one
    auto toBeIndexedNode = dataStorage_->node();
    for (const auto& indices : indexings_) {
      toBeIndexedNode = IndexNode::create(toBeIndexedNode, indices);
    }
    viewNode_ = toBeIndexedNode;
    oldDataNode_ = dataStorage_->node();
  }

 public:
  SharedData(NodePtr dataNode)
      : SharedData(std::make_shared<DataStorage>(dataNode), {}) {}

  SharedData(
      std::shared_ptr<DataStorage> dataStorage,
      std::vector<std::vector<Index>> indexings)
      : dataStorage_(dataStorage), indexings_(std::move(indexings)) {
    updateViewNodeIfNeeded();
  }

  void updateDataNode(NodePtr newNode) {
    if (!indexings_.empty()) {
      newNode =
          IndexedUpdateNode::create(dataStorage_->node(), indexings_, newNode);
    }
    dataStorage_->replaceNode(newNode);
  }

  // NOTE intended for optimizer
  void replaceNode(NodePtr newNode) {
    if (viewNode_.has_value()) {
      viewNode_ = newNode;
    } else {
      // graph optimization applies to all shallow copies
      dataStorage_->replaceNode(newNode);
    }
  }

  NodePtr getNode() {
    updateViewNodeIfNeeded();
    return viewNode_.value_or(dataStorage_->node());
  }

  std::shared_ptr<SharedData> applyIndices(std::vector<Index> indices) {
    std::vector<std::vector<Index>> newIndexings = this->indexings_;
    newIndexings.push_back(std::move(indices));
    return std::make_shared<SharedData>(dataStorage_, newIndexings);
  }
};

TensorBackend& JitTensorBase::wrappedBackend() const {
  return backend().wrappedBackend();
}

Optimizer& JitTensorBase::optimizer() const {
  return backend().optimizer();
}

Evaluator& JitTensorBase::evaluator() const {
  return backend().evaluator();
}

JitTensorBase::JitTensorBase(NodePtr node)
    : JitTensorBase(std::make_shared<SharedData>(node)) {}

JitTensorBase::JitTensorBase(std::shared_ptr<SharedData> sharedData)
    : sharedData_(sharedData) {}

JitTensorBase::~JitTensorBase() {}

const Tensor& JitTensorBase::getTensorOrEvalNode() const {
  if (!node()->getResult().has_value()) {
    eval();
  }
  return node()->getResult().value();
}

Tensor JitTensorBase::fromDataNode(NodePtr node) const {
  return fromSharedData(std::make_shared<SharedData>(node));
}

Tensor JitTensorBase::copy() {
  // Since a node's computation result is immutable, copy is free.
  return Tensor(clone());
}

Tensor JitTensorBase::shallowCopy() {
  // NOTE IR-captured computation semantics is immutable
  return fromSharedData(sharedData_);
}

TensorBackendType JitTensorBase::backendType() const {
  return TensorBackendType::Jit;
}

const Shape& JitTensorBase::shape() {
  return node()->shape();
}

fl::dtype JitTensorBase::type() {
  // TODO add type inference, which will
  // 1. Enable redundant cast removal -- this isn't easy to implement in the
  //    underlying backend (like OneDnnBackend) because in eager mode, we must
  //    make a copy even if type cast is redundant.
  // 2. avoid unnecessary materialization here, which enables more
  //    optimizations, e.g., fusion.
  return getTensorOrEvalNode().type();
}

bool JitTensorBase::isSparse() {
  return getTensorOrEvalNode().isSparse();
}

Location JitTensorBase::location() {
  // TODO keep track of location to avoid materialization
  return getTensorOrEvalNode().location();
}

void JitTensorBase::scalar(void* out) {
  // TODO support tensor.getAdapterBase() so we can use its `scalar` directly
  const auto& tensor = getTensorOrEvalNode();
  switch (type()) {
    case dtype::f16:
      throw std::runtime_error("[JitTensorBase::scalar] f16 unsupported");
    case dtype::f32:
      *((float*)out) = tensor.scalar<float>();
      return;
    case dtype::f64:
      *((double*)out) = tensor.scalar<double>();
      return;
    case dtype::b8:
      *((char*)out) = tensor.scalar<char>();
      return;
    case dtype::s16:
      *((short*)out) = tensor.scalar<short>();
      return;
    case dtype::s32:
      *((int*)out) = tensor.scalar<int>();
      return;
    case dtype::s64:
      *((long long*)out) = tensor.scalar<long long>();
      return;
    case dtype::u8:
      *((unsigned char*)out) = tensor.scalar<unsigned char>();
      return;
    case dtype::u16:
      *((unsigned short*)out) = tensor.scalar<unsigned short>();
      return;
    case dtype::u32:
      *((unsigned int*)out) = tensor.scalar<unsigned int>();
      return;
    case dtype::u64:
      *((unsigned long long*)out) = tensor.scalar<unsigned long long>();
      return;
  }
  throw std::runtime_error("[JitTensorBase::scalar] Unknown data type");
}

void JitTensorBase::device(void** out) {
  getTensorOrEvalNode().device(out);
}

void JitTensorBase::host(void* out) {
  getTensorOrEvalNode().host(out);
}

void JitTensorBase::unlock() {
  getTensorOrEvalNode().unlock();
}

bool JitTensorBase::isLocked() {
  return getTensorOrEvalNode().isLocked();
}

bool JitTensorBase::isContiguous() {
  // TODO infer contiguity? This is vaguely relevant to another idea -- annotate
  // nodes with strides and layout to enable more optimizations.
  return getTensorOrEvalNode().isContiguous();
}

Shape JitTensorBase::strides() {
  return getTensorOrEvalNode().strides();
}

const Stream& JitTensorBase::stream() const {
  // TODO can we infer this from leaf nodes?
  return getTensorOrEvalNode().stream();
}

Tensor JitTensorBase::astype(const dtype type) {
  // TODO cast node after we support type inference, so we can eliminate
  // redundant cast.
  return fromDataNode(CustomNode::create(
      "astype", {this->node()}, Shape(this->shape()), [=](auto inputs) {
        return inputs.at(0)->astype(type);
      }));
}

Tensor JitTensorBase::index(const std::vector<Index>& indices) {
  return fromSharedData(sharedData_->applyIndices(indices));
}

Tensor JitTensorBase::flatten() const {
  return fromDataNode(CustomNode::create(
      "flatten",
      {this->node()},
      Shape({node()->shape().elements()}),
      [=](auto inputs) { return inputs.at(0)->flatten(); }));
}

Tensor JitTensorBase::flat(const Index& idx) const {
  // TODO shape inference for custom node
  const auto& thisTensorResult =
      const_cast<JitTensorBase*>(this)->getTensorOrEvalNode();
  if (idx.type() == detail::IndexType::Tensor) {
    const auto& tensorIdx = idx.get<Tensor>();
    const auto& tensorIdxResult =
        toJitTensorBase(tensorIdx).getTensorOrEvalNode();
    return fromDataNode(
        ValueNode::create(thisTensorResult.flat(tensorIdxResult)));
  }
  return fromDataNode(ValueNode::create(thisTensorResult.flat(idx)));
}

Tensor JitTensorBase::asContiguousTensor() {
  // TODO add a node for this if we support contiguity or stride inference, so
  // we can eliminate redundant asContiguousTensor call.
  return fromDataNode(CustomNode::create(
      "asContiguousTensor", {this->node()}, Shape(shape()), [=](auto inputs) {
        return inputs.at(0)->asContiguousTensor();
      }));
}

void JitTensorBase::setContext(void* /* context */) {
  // no-op
}

void* JitTensorBase::getContext() {
  return nullptr;
}

std::string JitTensorBase::toString() {
  return getTensorOrEvalNode().toString();
}

std::ostream& JitTensorBase::operator<<(std::ostream& ostr) {
  ostr << toString();
  return ostr;
}

/******************** Assignment Operators ********************/
// NOTE Think SSA:
// x += 42
// ......
// --->
// x' = x + 42
// ...... (uses of x becomes x')
void JitTensorBase::assign(const Tensor& other) {
  sharedData_->updateDataNode(toJitTensorBase(other).node());
}

#define FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, TYPE)                \
  void JitTensorBase::OP(const TYPE& scalar) {                  \
    const auto dtype = dtype_traits<TYPE>::ctype;               \
    this->assign(backend().full(this->shape(), scalar, dtype)); \
  }

#define FL_JIT_TENSOR_ASSIGN_BINOP_TENSOR(OP, BINOP) \
  void JitTensorBase::OP(const Tensor& other) {      \
    this->assign(this->shallowCopy() BINOP other);   \
  }

#define FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, TYPE) \
  void JitTensorBase::OP(const TYPE& scalar) {             \
    this->assign(this->shallowCopy() BINOP scalar);        \
  }

#define FL_JIT_TENSOR_ASSIGN_OP(OP)                   \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, double);         \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, float);          \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, int);            \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned);       \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, bool);           \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, char);           \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned char);  \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, short);          \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned short); \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, long);           \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned long);  \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, long long);      \
  FL_JIT_TENSOR_ASSIGN_OP_SCALAR(OP, unsigned long long);

#define FL_JIT_TENSOR_ASSIGN_BINOP(OP, BINOP)                   \
  FL_JIT_TENSOR_ASSIGN_BINOP_TENSOR(OP, BINOP);                 \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, double);         \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, float);          \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, int);            \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned);       \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, bool);           \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, char);           \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned char);  \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, short);          \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned short); \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, long);           \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned long);  \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, long long);      \
  FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR(OP, BINOP, unsigned long long);

FL_JIT_TENSOR_ASSIGN_OP(assign); // =
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceAdd, +); // +=
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceSubtract, -); // -=
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceMultiply, *); // *=
FL_JIT_TENSOR_ASSIGN_BINOP(inPlaceDivide, /); // /=
#undef FL_JIT_TENSOR_ASSIGN_OP_SCALAR
#undef FL_JIT_TENSOR_ASSIGN_BINOP_TENSOR
#undef FL_JIT_TENSOR_ASSIGN_BINOP_SCALAR
#undef FL_JIT_TENSOR_ASSIGN_OP
#undef FL_JIT_TENSOR_ASSIGN_BINOP

NodePtr JitTensorBase::node() const {
  return sharedData_->getNode();
}

void JitTensorBase::eval() const {
  if (!node()->getResult().has_value()) {
    sharedData_->replaceNode(optimizer().optimize(node()));
    evaluator().eval(node());
  }
}

const JitTensorBase& toJitTensorBase(const Tensor& tensor) {
  return toJitTensorBase(const_cast<Tensor&>(tensor));
}

JitTensorBase& toJitTensorBase(Tensor& tensor) {
  auto type = tensor.backendType();
  if (type != TensorBackendType::Jit) {
    std::ostringstream oss;
    oss << "[toJitTensorBase] expected JIT-backed tensor, got " << type;
    throw std::invalid_argument(oss.str());
  }
  return tensor.getAdapter<JitTensorBase>();
}

} // namespace fl
