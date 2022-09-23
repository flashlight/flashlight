/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/trace/TracerTensorBase.h"
#include "flashlight/fl/tensor/TensorBase.h"

#define FL_TRACER_TENSOR_UNIMPLEMENTED \
  throw std::invalid_argument(         \
      "TracerTensorBase::" + std::string(__func__) + " - unimplemented.");

namespace fl {

std::unique_ptr<TensorAdapterBase> TracerTensorBase::clone() const {
  auto builder = TracerBase::TraceData::build(*backend().tracer(), __func__);
  builder->setInputs({{"input", tracedTensor_}});
  Tensor t = tracedTensor_; // impl-defined copy
  Tensor clonedTracedTensor = backend().toTracedTensor(std::move(t));
  builder->setOutputs({{"result", clonedTracedTensor}});
  backend().trace(builder->build());
  return detail::releaseAdapter(std::move(clonedTracedTensor));
}

Tensor TracerTensorBase::copy() {
  auto builder = TracerBase::TraceData::build(*backend().tracer(), __func__);
  builder->setInputs({{"input", tracedTensor_}});
  Tensor result = tracedTensor_.copy();
  builder->setOutputs({{"result", result}});
  backend().trace(builder->build());
  return backend().toTracedTensor(std::move(result));
}

Tensor TracerTensorBase::shallowCopy() {
  Tensor result = tracedTensor_.shallowCopy();

  backend().trace(
      "shallowCopy", {}, {{"input", tracedTensor_}}, {{"result", result}});

  return backend().toTracedTensor(std::move(result));
}

TensorBackendType TracerTensorBase::backendType() const {
  return TensorBackendType::Tracer;
}

const Shape& TracerTensorBase::shape() {
  return tracedTensor_.shape();
}

fl::dtype TracerTensorBase::type() {
  return tracedTensor_.type();
}

bool TracerTensorBase::isSparse() {
  return tracedTensor_.isSparse();
}

Location TracerTensorBase::location() {
  return tracedTensor_.location();
}

void TracerTensorBase::scalar(void* out) {
  switch (type()) {
    case dtype::f16:
      throw std::runtime_error("[TracerTensorBase::scalar] f16 unsupported");
    case dtype::f32:
      *((float*)out) = tracedTensor_.scalar<float>();
      break;
    case dtype::f64:
      *((double*)out) = tracedTensor_.scalar<double>();
      break;
    case dtype::b8:
      *((char*)out) = tracedTensor_.scalar<char>();
      break;
    case dtype::s16:
      *((short*)out) = tracedTensor_.scalar<short>();
      break;
    case dtype::s32:
      *((int*)out) = tracedTensor_.scalar<int>();
      break;
    case dtype::s64:
      *((long long*)out) = tracedTensor_.scalar<long long>();
      break;
    case dtype::u8:
      *((unsigned char*)out) = tracedTensor_.scalar<unsigned char>();
      break;
    case dtype::u16:
      *((unsigned short*)out) = tracedTensor_.scalar<unsigned short>();
      break;
    case dtype::u32:
      *((unsigned int*)out) = tracedTensor_.scalar<unsigned int>();
      break;
    case dtype::u64:
      *((unsigned long long*)out) = tracedTensor_.scalar<unsigned long long>();
      break;
    default:
      throw std::runtime_error("Type not supported");
  }

  return;
}

void TracerTensorBase::device(void** out) {
  tracedTensor_.device(out);
}

void TracerTensorBase::host(void* out) {
  tracedTensor_.host(out);
}

void TracerTensorBase::unlock() {
  tracedTensor_.unlock();
}

bool TracerTensorBase::isLocked() {
  return tracedTensor_.isLocked();
}

bool TracerTensorBase::isContiguous() {
  return tracedTensor_.isContiguous();
}

Shape TracerTensorBase::strides() {
  return tracedTensor_.strides();
}

const Stream& TracerTensorBase::stream() const {
  return tracedTensor_.stream();
}

Tensor TracerTensorBase::astype(const dtype type) {
  auto builder = TracerBase::TraceData::build(*backend().tracer(), __func__);
  builder->setArgs({{"type", type}})->setInputs({{"input", tracedTensor_}});
  Tensor result = tracedTensor_.astype(type);
  builder->setOutputs({{"result", result}});
  backend().trace(builder->build());
  return backend().toTracedTensor(std::move(result));
  ;
}

Tensor TracerTensorBase::index(const std::vector<Index>& indices) {
  auto builder = TracerBase::TraceData::build(*backend().tracer(), __func__);
  builder->setArgs({{"indices", indices}})
      ->setInputs({{"input", tracedTensor_}});
  Tensor result = tracedTensor_(indices);
  builder->setOutputs({{"result", result}});
  backend().trace(builder->build());
  return backend().toTracedTensor(std::move(result));
}

Tensor TracerTensorBase::flatten() const {
  auto builder = TracerBase::TraceData::build(*backend().tracer(), __func__);
  builder->setInputs({{"input", tracedTensor_}});
  Tensor result = tracedTensor_.flatten();
  builder->setOutputs({{"result", result}});
  backend().trace(builder->build());
  return backend().toTracedTensor(std::move(result));
}

Tensor TracerTensorBase::flat(const Index& idx) const {
  auto builder = TracerBase::TraceData::build(*backend().tracer(), __func__);
  builder->setInputs({{"input", tracedTensor_}});
  Tensor result = tracedTensor_.flat(idx);
  builder->setOutputs({{"result", result}});
  backend().trace(builder->build());
  return backend().toTracedTensor(std::move(result));
}

Tensor TracerTensorBase::asContiguousTensor() {
  auto builder = TracerBase::TraceData::build(*backend().tracer(), __func__);
  builder->setInputs({{"input", tracedTensor_}});
  Tensor result = tracedTensor_.asContiguousTensor();
  builder->setOutputs({{"result", result}});
  backend().trace(builder->build());
  return backend().toTracedTensor(std::move(result));
}

void TracerTensorBase::setContext(void* context) {
  tracedTensor_.setContext(context);
}

void* TracerTensorBase::getContext() {
  return tracedTensor_.getContext();
}

std::string TracerTensorBase::toString() {
  backend().trace("toString", {}, {{"input", tracedTensor_}}, {});
  return tracedTensor_.toString();
}

std::ostream& TracerTensorBase::operator<<(std::ostream& ostr) {
  backend().trace("operator<<", {}, {{"input", tracedTensor_}}, {});
  return tracedTensor_.operator<<(ostr);
}

/******************** Assignment Operators ********************/
#define FL_TRACER_TENSOR_ASSIGN_OP_TYPE(OP, FUN, TYPE)                  \
  void TracerTensorBase::OP(const TYPE& val) {                          \
    backend().trace(#OP, {}, {{"lhs", tracedTensor_}}, {{"lhs", val}}); \
    tracedTensor_.FUN(val);                                             \
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
