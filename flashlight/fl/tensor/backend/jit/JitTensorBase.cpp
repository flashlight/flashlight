/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"

#include <memory>
#include <sstream>
#include <stdexcept>

#define FL_JIT_TENSOR_UNIMPLEMENTED \
  throw std::invalid_argument(      \
      "JitTensorBase::" + std::string(__func__) + " - unimplemented.");

namespace fl {

// represents the data referred to by indexinges
struct DataStorage {
  Node* node;

  DataStorage(Node* node) : node(node) {
    node->incRefCount(); // shallow copies counts as 1 use
  }

  ~DataStorage() {
    node->decRefCount();
  }

  void replaceNode(Node* newNode) {
    newNode->incRefCount();
    node->decRefCount();
    node = newNode;
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
  std::optional<Node*> oldDataNode_{std::nullopt}; // None iff no indexings
  std::optional<Node*> viewNode_{std::nullopt}; // None iff no indexings

  void updateViewNodeIfNeeded() {
    if (indexings_.empty() || dataStorage_->node == oldDataNode_) {
      return;
    }
    if (viewNode_.has_value()) {
      viewNode_.value()->decRefCount();
    }
    // apply index one by one
    auto toBeIndexedNode = dataStorage_->node;
    for (const auto& indices : indexings_) {
      toBeIndexedNode = IndexNode::create(toBeIndexedNode, indices);
    }
    toBeIndexedNode->incRefCount();
    viewNode_ = toBeIndexedNode;
    oldDataNode_ = dataStorage_->node;
  }

 public:
  SharedData(Node* dataNode)
      : SharedData(std::make_shared<DataStorage>(dataNode), {}) {}

  SharedData(
      std::shared_ptr<DataStorage> dataStorage,
      std::vector<std::vector<Index>> indexings)
      : dataStorage_(dataStorage), indexings_(std::move(indexings)) {
    updateViewNodeIfNeeded();
  }

  ~SharedData() {
    if (viewNode_.has_value()) {
      viewNode_.value()->decRefCount();
    }
  }

  void updateDataNode(Node* newNode) {
    if (viewNode_.has_value()) {
      throw std::runtime_error(
          "[SharedData::updateDataNode] Currently no support for indexed update");
    }
    dataStorage_->replaceNode(newNode);
  }

  // NOTE intended for optimizer
  void replaceNode(Node* newNode) {
    if (viewNode_.has_value()) {
      newNode->incRefCount();
      viewNode_.value()->decRefCount();
      viewNode_ = newNode;
    } else {
      // graph optimization applies to all shallow copies
      dataStorage_->replaceNode(newNode);
    }
  }

  Node* getNode() {
    updateViewNodeIfNeeded();
    return viewNode_.value_or(dataStorage_->node);
  }

  std::shared_ptr<SharedData> applyIndices(std::vector<Index> indices) {
    std::vector<std::vector<Index>> newIndexings = this->indexings_;
    newIndexings.push_back(std::move(indices));
    return std::make_shared<SharedData>(dataStorage_, newIndexings);
  }
};

JitTensorBase::JitTensorBase(Node* node)
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
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

bool JitTensorBase::isSparse() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

Location JitTensorBase::location() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

void JitTensorBase::scalar(void* /* out */) {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

void JitTensorBase::device(void** /* out */) {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

void JitTensorBase::host(void* /* out */) {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

void JitTensorBase::unlock() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

bool JitTensorBase::isLocked() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

bool JitTensorBase::isContiguous() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

Shape JitTensorBase::strides() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

const Stream& JitTensorBase::stream() const {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

Tensor JitTensorBase::astype(const dtype /* type */) {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

Tensor JitTensorBase::index(const std::vector<Index>& indices) {
  return fromSharedData(sharedData_->applyIndices(indices));
}

Tensor JitTensorBase::flatten() const {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

Tensor JitTensorBase::flat(const Index& /* idx */) const {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

Tensor JitTensorBase::asContiguousTensor() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

void JitTensorBase::setContext(void* /* context */) {
  // Used to store arbitrary data on a Tensor - can be a noop.
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

void* JitTensorBase::getContext() {
  // Used to store arbitrary data on a Tensor - can be a noop.
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

std::string JitTensorBase::toString() {
  return getTensorOrEvalNode().toString();
}

std::ostream& JitTensorBase::operator<<(std::ostream& /* ostr */) {
  FL_JIT_TENSOR_UNIMPLEMENTED;
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

Node* JitTensorBase::node() const {
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
