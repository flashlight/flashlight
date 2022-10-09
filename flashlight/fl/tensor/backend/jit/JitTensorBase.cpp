/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"

#include <sstream>

#define FL_JIT_TENSOR_UNIMPLEMENTED \
  throw std::invalid_argument(      \
      "JitTensorBase::" + std::string(__func__) + " - unimplemented.");

namespace fl {

JitTensorBase::JitTensorBase(Node* node) : node_(node) {
  node_->incRefCount();
}

JitTensorBase::~JitTensorBase() {
  node_->decRefCount();
}

const Tensor& JitTensorBase::getTensorOrEvalNode() const {
  if (!node_->getResult().has_value()) {
    eval();
  }
  return node_->getResult().value();
}

Tensor JitTensorBase::copy() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
}

Tensor JitTensorBase::shallowCopy() {
  return fromNode(node_);
}

TensorBackendType JitTensorBase::backendType() const {
  return TensorBackendType::Jit;
}

const Shape& JitTensorBase::shape() {
  FL_JIT_TENSOR_UNIMPLEMENTED;
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

Tensor JitTensorBase::index(const std::vector<Index>& /* indices */) {
  FL_JIT_TENSOR_UNIMPLEMENTED;
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
#define FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, TYPE)           \
  void JitTensorBase::OP(const TYPE& /* val */) {             \
    throw std::invalid_argument(                              \
        "JitTensorBase::" + std::string(#OP) + " for type " + \
        std::string(#TYPE));                                  \
  }

#define FL_JIT_TENSOR_ASSIGN_OP_STUB(OP)                 \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, Tensor);         \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, double);         \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, float);          \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, int);            \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, unsigned);       \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, bool);           \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, char);           \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, unsigned char);  \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, short);          \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, unsigned short); \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, long);           \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, unsigned long);  \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, long long);      \
  FL_JIT_TENSOR_ASSIGN_OP_TYPE_STUB(OP, unsigned long long);

FL_JIT_TENSOR_ASSIGN_OP_STUB(assign); // =
FL_JIT_TENSOR_ASSIGN_OP_STUB(inPlaceAdd); // +=
FL_JIT_TENSOR_ASSIGN_OP_STUB(inPlaceSubtract); // -=
FL_JIT_TENSOR_ASSIGN_OP_STUB(inPlaceMultiply); // *=
FL_JIT_TENSOR_ASSIGN_OP_STUB(inPlaceDivide); // /=
#undef FL_JIT_TENSOR_ASSIGN_OP_TYPE
#undef FL_JIT_TENSOR_ASSIGN_OP

Node* JitTensorBase::node() const {
  return node_;
}

void JitTensorBase::eval() const {
  evaluator().eval(node_);
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
