/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/trace/TracerBackendBase.h"

#include <iostream>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/trace/TracerTensorBase.h"

namespace fl {
namespace {

Tensor& getTracedTensor(const Tensor& t) {
  if (t.backendType() != TensorBackendType::Tracer) {
    throw std::invalid_argument(
        "getTracedTensor: Tensor is not TracerTensor backed");
  }
  return t.getAdapter<TracerTensorBase>().tensor();
}

} // namespace

TensorBackendType TracerBackendBase::backendType() const {
  return TensorBackendType::Tracer;
}

/* -------------------------- Compute Functions -------------------------- */

void TracerBackendBase::eval(const Tensor& tensor) {
  Tensor& traced = getTracedTensor(tensor);
  trace(__func__, {{"tensor", traced}});
  backend().eval(traced);
}

bool TracerBackendBase::supportsDataType(const fl::dtype& dtype) const {
  return backend().supportsDataType(dtype);
}

void TracerBackendBase::getMemMgrInfo(
    const char* msg,
    const int deviceId,
    std::ostream* ostream) {
  backend().getMemMgrInfo(msg, deviceId, ostream);
}

void TracerBackendBase::setMemMgrLogStream(std::ostream* stream) {
  backend().setMemMgrLogStream(stream);
}

void TracerBackendBase::setMemMgrLoggingEnabled(const bool enabled) {
  backend().setMemMgrLoggingEnabled(enabled);
}

void TracerBackendBase::setMemMgrFlushInterval(const size_t interval) {
  backend().setMemMgrFlushInterval(interval);
}

/* -------------------------- Rand Functions -------------------------- */

void TracerBackendBase::setSeed(const int seed) {
  trace(__func__, {{"seed", seed}});
  backend().setSeed(seed);
}

Tensor TracerBackendBase::randn(const Shape& shape, dtype type) {
  Tensor result = backend().randn(shape, type);
  trace("randn", {}, {}, {{"result", result}});
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::rand(const Shape& shape, dtype type) {
  Tensor result = backend().rand(shape, type);
  trace("rand", {}, {}, {{"result", result}});
  return toTracedTensor(std::move(result));
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
#define FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(TYPE)                   \
  Tensor TracerBackendBase::fromScalar(TYPE value, const dtype type) { \
    Tensor result = backend().fromScalar(value, type);                 \
    trace(                                                             \
        "fromScalar",                                                  \
        {{"value", value}, {"type", type}},                            \
        {},                                                            \
        {{"result", result}});                                         \
    return toTracedTensor(std::move(result));                          \
  }                                                                    \
  Tensor TracerBackendBase::full(                                      \
      const Shape& dims, TYPE value, const dtype type) {               \
    Tensor result = backend().full(dims, value, type);                 \
    trace(                                                             \
        "full",                                                        \
        {{"dims", dims}, {"value", value}, {"type", type}},            \
        {},                                                            \
        {{"result", result}});                                         \
    return toTracedTensor(std::move(result));                          \
  }
FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(const double&);
FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(const float&);
FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(const int&);
FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned&);
FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(const char&);
FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned char&);
FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(const long&);
FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned long&);
FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(const long long&);
FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned long long&);
FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(const bool&);
FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(const short&);
FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned short&);

Tensor TracerBackendBase::identity(const Dim dim, const dtype type) {
  Tensor result = backend().identity(dim, type);
  trace("identity", {{"dim", dim}, {"type", type}}, {}, {{"result", result}});
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::arange(
    const Shape& shape,
    const Dim seqDim,
    const dtype type) {
  Tensor result = backend().arange(shape, seqDim, type);
  trace(
      "arange",
      {{"shape", shape}, {"seqDim", seqDim}, {"type", type}},
      {},
      {{"result", result}});
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::iota(
    const Shape& dims,
    const Shape& tileDims,
    const dtype type) {
  Tensor result = backend().iota(dims, tileDims, type);
  trace(
      "iota",
      {{"dims", dims}, {"tileDims", tileDims}, {"type", type}},
      {},
      {{"result", result}});
  return toTracedTensor(std::move(result));
}

/************************ Shaping and Indexing *************************/
Tensor TracerBackendBase::reshape(const Tensor& _tensor, const Shape& shape) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"shape", shape}})->setInputs({{"tensor", tensor}});
  Tensor result = backend().reshape(tensor, shape);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::transpose(
    const Tensor& _tensor,
    const Shape& axes /* = {} */) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axes", axes}})->setInputs({{"tensor", tensor}});
  Tensor result = backend().transpose(tensor, axes);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::tile(const Tensor& _tensor, const Shape& shape) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"shape", shape}})->setInputs({{"tensor", tensor}});
  Tensor result = backend().tile(tensor, shape);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::concatenate(
    const std::vector<Tensor>& tensors,
    const unsigned axis) {
  std::vector<Tensor> tracedTensors;
  for (const auto& tensor : tensors) {
    tracedTensors.push_back(getTracedTensor(tensor));
  }
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axis", axis}})->setInputs({{"tensors", tracedTensors}});
  Tensor result = backend().concatenate(tracedTensors, axis);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::nonzero(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().nonzero(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::pad(
    const Tensor& _input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"padWidths", padWidths}, {"type", type}})
      ->setInputs({{"input", input}});
  Tensor result = backend().pad(input, padWidths, type);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

/************************** Unary Operators ***************************/

Tensor TracerBackendBase::exp(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().exp(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::log(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().log(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::negative(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().negative(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::logicalNot(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().logicalNot(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::log1p(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().log1p(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::sin(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().sin(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::cos(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().cos(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::sqrt(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().sqrt(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::tanh(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().tanh(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::floor(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().floor(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::ceil(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().ceil(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::rint(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().rint(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::absolute(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().absolute(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::sigmoid(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().sigmoid(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::erf(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().erf(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::flip(const Tensor& _tensor, const unsigned dim) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().flip(tensor, dim);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::clip(
    const Tensor& _tensor,
    const Tensor& low,
    const Tensor& high) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"low", low}, {"high", high}})
      ->setInputs({{"tensor", tensor}});
  Tensor result = backend().clip(tensor, low, high);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::clip(
    const Tensor& _tensor,
    const double& low,
    const double& high) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"low", low}, {"high", high}})
      ->setInputs({{"tensor", tensor}});
  Tensor result = backend().clip(tensor, low, high);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::roll(
    const Tensor& _tensor,
    const int shift,
    const unsigned axis) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"shift", shift}, {"axis", axis}})
      ->setInputs({{"tensor", tensor}});
  Tensor result = backend().roll(tensor, shift, axis);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::isnan(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().isnan(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::isinf(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().isinf(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::sign(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().sign(getTracedTensor(tensor));
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::tril(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().tril(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::triu(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"tensor", tensor}});
  Tensor result = backend().triu(tensor);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::where(
    const Tensor& _condition,
    const Tensor& x,
    const Tensor& y) {
  Tensor& condition = getTracedTensor(_condition);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"condition", condition}, {"x", x}, {"y", y}});
  Tensor result = backend().where(condition, x, y);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

void TracerBackendBase::topk(
    Tensor& _values,
    Tensor& _indices,
    const Tensor& _input,
    const unsigned k,
    const Dim axis,
    const SortMode sortMode /* = SortMode::Descending */) {
  Tensor& values = getTracedTensor(_values);
  Tensor& indices = getTracedTensor(_indices);
  Tensor& input = getTracedTensor(_input);

  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs(
      {{"input", input}, {"k", k}, {"axis", axis}, {"sortMode", sortMode}});
  builder->setInputs({{"input", input}});
  backend().topk(values, indices, input, k, axis, sortMode);
  builder->setOutputs({{"values", values}, {"indices", indices}});
  trace(builder->build());
}

Tensor TracerBackendBase::sort(
    const Tensor& _input,
    const Dim axis,
    const SortMode sortMode) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axis", axis}, {"sortMode", sortMode}})
      ->setInputs({{"input", input}});
  Tensor result = backend().sort(input, axis, sortMode);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

void TracerBackendBase::sort(
    Tensor& _values,
    Tensor& _indices,
    const Tensor& _input,
    const Dim axis,
    const SortMode sortMode /* = SortMode::Descending */) {
  Tensor& values = getTracedTensor(_values);
  Tensor& indices = getTracedTensor(_indices);
  Tensor& input = getTracedTensor(_input);

  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axis", axis}, {"sortMode", sortMode}})
      ->setInputs({{"input", input}});
  backend().sort(values, indices, input, axis, sortMode);
  builder->setOutputs({{"values", values}, {"indices", indices}});
  trace(builder->build());
}

Tensor TracerBackendBase::argsort(
    const Tensor& _input,
    const Dim axis,
    const SortMode sortMode) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axis", axis}, {"sortMode", sortMode}})
      ->setInputs({{"input", input}});
  Tensor result = backend().argsort(input, axis, sortMode);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

/************************** Binary Operators ***************************/
#define FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, TYPE)               \
  Tensor TracerBackendBase::FUNC(const Tensor& _lhs, TYPE rhs) {   \
    auto builder = TracerBase::TraceData::build(*tracer(), #FUNC); \
    Tensor& lhs = getTracedTensor(_lhs);                           \
    builder->setInputs({{"lhs", lhs}, {"rhs", rhs}});              \
    Tensor result = lhs OP rhs;                                    \
    builder->setOutputs({{"result", result}});                     \
    trace(builder->build());                                       \
    return toTracedTensor(std::move(result));                      \
  }                                                                \
  Tensor TracerBackendBase::FUNC(TYPE lhs, const Tensor& _rhs) {   \
    auto builder = TracerBase::TraceData::build(*tracer(), #FUNC); \
    Tensor& rhs = getTracedTensor(_rhs);                           \
    builder->setInputs({{"lhs", lhs}, {"rhs", rhs}});              \
    Tensor result = lhs OP rhs;                                    \
    builder->setOutputs({{"result", result}});                     \
    trace(builder->build());                                       \
    return toTracedTensor(std::move(result));                      \
  }

#define FL_TRACER_BINARY_OP_LITERALS_DEF(FUNC, OP)                   \
  FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, const bool&);               \
  FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, const int&);                \
  FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned&);           \
  FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, const char&);               \
  FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned char&);      \
  FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, const long&);               \
  FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned long&);      \
  FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, const long long&);          \
  FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned long long&); \
  FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, const double&);             \
  FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, const float&);              \
  FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, const short&);              \
  FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned short&);

// Operations on fl::Tensor call the respective operator overloads that are
// already defined on af::arrays
#define FL_TRACER_BINARY_OP_DEF(OP, FUNC)                                  \
  Tensor TracerBackendBase::FUNC(const Tensor& _lhs, const Tensor& _rhs) { \
    Tensor& lhs = getTracedTensor(_lhs);                                   \
    Tensor& rhs = getTracedTensor(_rhs);                                   \
    auto builder = TracerBase::TraceData::build(*tracer(), #FUNC);         \
    builder->setInputs({{"lhs", lhs}, {"rhs", rhs}});                      \
    Tensor result = lhs OP rhs;                                            \
    builder->setOutputs({{"result", result}});                             \
    trace(builder->build());                                               \
    return toTracedTensor(std::move(result));                              \
  }                                                                        \
  FL_TRACER_BINARY_OP_LITERALS_DEF(FUNC, OP);

// Definitions
// Since ArrayFire implements operator overloads, map both fl::Tensor
// functions and fl::Tensor operator overloads back to the af::array
// overloads.
FL_TRACER_BINARY_OP_DEF(+, add);
FL_TRACER_BINARY_OP_DEF(-, sub);
FL_TRACER_BINARY_OP_DEF(*, mul);
FL_TRACER_BINARY_OP_DEF(/, div);
FL_TRACER_BINARY_OP_DEF(==, eq);
FL_TRACER_BINARY_OP_DEF(!=, neq);
FL_TRACER_BINARY_OP_DEF(<, lessThan);
FL_TRACER_BINARY_OP_DEF(<=, lessThanEqual);
FL_TRACER_BINARY_OP_DEF(>, greaterThan);
FL_TRACER_BINARY_OP_DEF(>=, greaterThanEqual);
FL_TRACER_BINARY_OP_DEF(||, logicalOr);
FL_TRACER_BINARY_OP_DEF(&&, logicalAnd);
FL_TRACER_BINARY_OP_DEF(%, mod);
FL_TRACER_BINARY_OP_DEF(&, bitwiseAnd);
FL_TRACER_BINARY_OP_DEF(|, bitwiseOr);
FL_TRACER_BINARY_OP_DEF(^, bitwiseXor);
FL_TRACER_BINARY_OP_DEF(<<, lShift);
FL_TRACER_BINARY_OP_DEF(>>, rShift);
#undef FL_TRACER_BINARY_OP_DEF
#undef FL_TRACER_BINARY_OP_TYPE_DEF
#undef FL_TRACER_BINARY_OP_LITERALS_DEF

Tensor TracerBackendBase::minimum(const Tensor& _lhs, const Tensor& _rhs) {
  Tensor& lhs = getTracedTensor(_lhs);
  Tensor& rhs = getTracedTensor(_rhs);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"lhs", lhs}, {"rhs", rhs}});
  Tensor result = backend().minimum(lhs, rhs);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::maximum(const Tensor& _lhs, const Tensor& _rhs) {
  Tensor& lhs = getTracedTensor(_lhs);
  Tensor& rhs = getTracedTensor(_rhs);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"lhs", lhs}, {"rhs", rhs}});
  Tensor result = lhs.backend().maximum(lhs, rhs);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::power(const Tensor& _lhs, const Tensor& _rhs) {
  Tensor& lhs = getTracedTensor(_lhs);
  Tensor& rhs = getTracedTensor(_rhs);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setInputs({{"lhs", lhs}, {"rhs", rhs}});
  Tensor result = lhs.backend().power(lhs, rhs);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

/************************** BLAS ***************************/

Tensor TracerBackendBase::matmul(
    const Tensor& _lhs,
    const Tensor& _rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp) {
  Tensor lhs = getTracedTensor(_lhs);
  Tensor rhs = getTracedTensor(_rhs);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"lhsProp", lhsProp}, {"rhsProp", rhsProp}})
      ->setInputs({{"lhs", lhs}, {"rhs", rhs}});
  Tensor result = backend().matmul(lhs, rhs, lhsProp, rhsProp);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}
/************************** Reductions ***************************/

Tensor TracerBackendBase::amin(
    const Tensor& _input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axes", axes}, {"keepDims", keepDims}})
      ->setInputs({{"input", input}});
  Tensor result = input.backend().amin(input, axes, keepDims);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::amax(
    const Tensor& _input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axes", axes}, {"keepDims", keepDims}})
      ->setInputs({{"input", input}});
  Tensor result = input.backend().amax(input, axes, keepDims);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

void TracerBackendBase::min(
    Tensor& _values,
    Tensor& _indices,
    const Tensor& _input,
    const unsigned axis,
    const bool keepDims) {
  Tensor& values = getTracedTensor(_values);
  Tensor& indices = getTracedTensor(_indices);
  Tensor& input = getTracedTensor(_input);

  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"input", input}, {"axis", axis}, {"keepDims", keepDims}});
  builder->setInputs({{"input", input}});
  input.backend().min(values, indices, input, axis, keepDims);
  builder->setOutputs({{"values", values}, {"indices", indices}});
  trace(builder->build());
}

void TracerBackendBase::max(
    Tensor& _values,
    Tensor& _indices,
    const Tensor& _input,
    const unsigned axis,
    const bool keepDims /* = false */) {
  Tensor& values = getTracedTensor(_values);
  Tensor& indices = getTracedTensor(_indices);
  Tensor& input = getTracedTensor(_input);

  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"input", input}, {"axis", axis}, {"keepDims", keepDims}});
  builder->setInputs({{"input", input}});
  input.backend().max(values, indices, input, axis, keepDims);
  builder->setOutputs({{"values", values}, {"indices", indices}});
  trace(builder->build());
  input.backend().max(values, indices, input, axis, keepDims);
}

Tensor TracerBackendBase::sum(
    const Tensor& _input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axes", axes}, {"keepDims", keepDims}})
      ->setInputs({{"input", input}});
  Tensor result = input.backend().sum(input, axes, keepDims);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::cumsum(const Tensor& _input, const unsigned axis) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axis", axis}})->setInputs({{"input", input}});
  Tensor result = input.backend().cumsum(input, axis);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::argmax(
    const Tensor& _input,
    const unsigned axis,
    const bool keepDims /* = false */) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axis", axis}, {"keepDims", keepDims}})
      ->setInputs({{"input", input}});
  Tensor result = input.backend().argmax(input, axis, keepDims);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::argmin(
    const Tensor& _input,
    const unsigned axis,
    const bool keepDims /* = false */) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axis", axis}, {"keepDims", keepDims}})
      ->setInputs({{"input", input}});
  Tensor result = input.backend().argmin(input, axis, keepDims);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::mean(
    const Tensor& _input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axes", axes}, {"keepDims", keepDims}})
      ->setInputs({{"input", input}});
  Tensor result = input.backend().mean(input, axes, keepDims);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::median(
    const Tensor& _input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axes", axes}, {"keepDims", keepDims}})
      ->setInputs({{"input", input}});
  Tensor result = input.backend().median(input, axes, keepDims);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::var(
    const Tensor& _input,
    const std::vector<int>& axes /* = {} */,
    const bool bias,
    const bool keepDims /* = false */) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axes", axes}, {"bias", bias}, {"keepDims", keepDims}})
      ->setInputs({{"input", input}});
  Tensor result = input.backend().var(input, axes, bias, keepDims);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::std(
    const Tensor& _input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axes", axes}, {"keepDims", keepDims}})
      ->setInputs({{"input", input}});
  Tensor result = input.backend().std(input, axes, keepDims);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::norm(
    const Tensor& _input,
    const std::vector<int>& axes /* = {} */,
    double p /* = 2 */,
    const bool keepDims /* = false */) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axes", axes}, {"p", p}, {"keepDims", keepDims}})
      ->setInputs({{"input", input}});
  Tensor result = input.backend().norm(input, axes, p, keepDims);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::countNonzero(
    const Tensor& _input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axes", axes}, {"keepDims", keepDims}})
      ->setInputs({{"input", input}});
  Tensor result = input.backend().countNonzero(input, axes, keepDims);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::any(
    const Tensor& _input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axes", axes}, {"keepDims", keepDims}})
      ->setInputs({{"input", input}});
  Tensor result = input.backend().any(input, axes, keepDims);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

Tensor TracerBackendBase::all(
    const Tensor& _input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor& input = getTracedTensor(_input);
  auto builder = TracerBase::TraceData::build(*tracer(), __func__);
  builder->setArgs({{"axes", axes}, {"keepDims", keepDims}})
      ->setInputs({{"input", input}});
  Tensor result = input.backend().all(input, axes, keepDims);
  builder->setOutputs({{"result", result}});
  trace(builder->build());
  return toTracedTensor(std::move(result));
}

void TracerBackendBase::print(const Tensor& _tensor) {
  Tensor& tensor = getTracedTensor(_tensor);
  tensor.backend().print(tensor);
}

} // namespace fl
