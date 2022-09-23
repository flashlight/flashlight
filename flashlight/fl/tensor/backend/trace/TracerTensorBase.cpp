/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/trace/TracerTensorBase.h"

#define FL_STUB_TENSOR_UNIMPLEMENTED \
  throw std::invalid_argument(       \
      "TracerTensorBase::" + std::string(__func__) + " - unimplemented.");

namespace fl {

// void TensorBackendBase::trace(){backend().trace()}

TracerTensorBase::TracerTensorBase() {}

TracerTensorBase::TracerTensorBase(
    const Shape& /* shape */,
    fl::dtype /* type */,
    const void* /* ptr */,
    Location /* memoryLocation */) {}

TracerTensorBase::TracerTensorBase(
    const Dim /* nRows */,
    const Dim /* nCols */,
    const Tensor& /* values */,
    const Tensor& /* rowIdx */,
    const Tensor& /* colIdx */,
    StorageType /* storageType */) {}

std::unique_ptr<TensorAdapterBase> TracerTensorBase::clone() const {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

Tensor TracerTensorBase::copy() {
  Tensor result = tracedTensor_->copy();
  backend().trace(
      "copy", {}, {{"input", *tracedTensor_}}, {{"result", result}});
  return result;
}

Tensor TracerTensorBase::shallowCopy() {
  Tensor result = tracedTensor_->shallowCopy();
  backend().trace(
      "shallowCopy", {}, {{"input", *tracedTensor_}}, {{"result", result}});
  return result;
}

TensorBackendType TracerTensorBase::backendType() const {
  return TensorBackendType::Tracer;
}

const Shape& TracerTensorBase::shape() {
  return tracedTensor_->shape();
}

fl::dtype TracerTensorBase::type() {
  return tracedTensor_->type();
}

bool TracerTensorBase::isSparse() {
  return tracedTensor_->isSparse();
}

Location TracerTensorBase::location() {
  return tracedTensor_->location();
}

void TracerTensorBase::scalar(void* out) {
  const auto& tensor = *tracedTensor_;
  switch (type()) {
    case dtype::f16:
      throw std::runtime_error("[TracerTensorBase::scalar] f16 unsupported");
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
  throw std::runtime_error("[TracerTensorBase::scalar] Unknown data type");
}

void TracerTensorBase::device(void** out) {
  tracedTensor_->device(out);
}

void TracerTensorBase::host(void* out) {
  tracedTensor_->host(out);
}

void TracerTensorBase::unlock() {
  tracedTensor_->unlock();
}

bool TracerTensorBase::isLocked() {
  return tracedTensor_->isLocked();
}

bool TracerTensorBase::isContiguous() {
  return tracedTensor_->isContiguous();
}

Shape TracerTensorBase::strides() {
  return tracedTensor_->strides();
}

const Stream& TracerTensorBase::stream() const {
  return tracedTensor_->stream();
}

Tensor TracerTensorBase::astype(const dtype type) {
  Tensor result = tracedTensor_->astype(type);
  backend().trace(
      "astype",
      {{"type", type}},
      {{"input", *tracedTensor_}},
      {{"result", result}});
  return result;
}

Tensor TracerTensorBase::index(const std::vector<Index>& indices) {
  Tensor result = (*tracedTensor_)(indices);
  backend().trace(
      "index",
      {{"indices", indices}},
      {{"input", *tracedTensor_}},
      {{"result", result}});
  return result;
}

Tensor TracerTensorBase::flatten() const {
  Tensor result = tracedTensor_->flatten();
  backend().trace(
      "flatten", {}, {{"input", *tracedTensor_}}, {{"result", result}});
  return result;
}

Tensor TracerTensorBase::flat(const Index& idx) const {
  Tensor result = tracedTensor_->flat(idx);
  backend().trace(
      "flat",
      {},
      {{"input", *tracedTensor_}, {"index", idx}},
      {{"result", result}});
  return result;
}

Tensor TracerTensorBase::asContiguousTensor() {
  Tensor result = tracedTensor_->asContiguousTensor();
  backend().trace(
      "asContiguousTensor",
      {},
      {{"input", *tracedTensor_}},
      {{"result", result}});
  return result;
}

void TracerTensorBase::setContext(void* context) {
  tracedTensor_->setContext(context);
}

void* TracerTensorBase::getContext() {
  return tracedTensor_->getContext();
}

std::string TracerTensorBase::toString() {
  backend().trace("toString", {}, {{"input", *tracedTensor_}}, {});
  return tracedTensor_->toString();
}

std::ostream& TracerTensorBase::operator<<(std::ostream& ostr) {
  backend().trace("operator<<", {}, {{"input", *tracedTensor_}}, {});
  return tracedTensor_->operator<<(ostr);
}

/******************** Assignment Operators ********************/
#define FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, TYPE)                   \
  void TracerTensorBase::OP(const TYPE& val) {                           \
    backend().trace(#OP, {}, {{"lhs", *tracedTensor_}}, {{"lhs", val}}); \
    tracedTensor_->FUN(val);                                             \
  }

#define FL_TRACER_TENSOR_ASSIGN_OP_LITERALS(OP, FUN)        \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, double);         \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, float);          \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, int);            \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, unsigned);       \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, bool);           \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, char);           \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, unsigned char);  \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, short);          \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, unsigned short); \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, long);           \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, unsigned long);  \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, long long);      \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, unsigned long long);

#define FL_TRACER_TENSOR_ASSIGN_OP(OP, FUN)         \
  FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, Tensor); \
  FL_TRACER_TENSOR_ASSIGN_OP_LITERALS(OP, FUN)

FL_TRACER_TENSOR_ASSIGN_OP(assign, operator=); // =
FL_TRACER_TENSOR_ASSIGN_OP(inPlaceAdd, operator+=); // +=
FL_TRACER_TENSOR_ASSIGN_OP(inPlaceSubtract, operator-=); // -=
FL_TRACER_TENSOR_ASSIGN_OP(inPlaceMultiply, operator*=); // *=
FL_TRACER_TENSOR_ASSIGN_OP(inPlaceDivide, operator/=); // /=
#undef FL_TRACER_TENSOR_ASSIGN_OP_TYPE
#undef FL_TRACER_TENSOR_ASSIGN_OP

} // namespace fl
