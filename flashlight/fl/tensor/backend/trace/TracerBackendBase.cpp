/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/trace/TracerBackendBase.h"

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

TensorBackendType TracerBackendBase::backendType() const {
  return TensorBackendType::Tracer;
}

/* -------------------------- Compute Functions -------------------------- */

void TracerBackendBase::eval(const Tensor& tensor) {
  trace(__func__, {{"tensor", tensor}});
  backend().eval(tensor);
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
  return result;
}

Tensor TracerBackendBase::rand(const Shape& shape, dtype type) {
  Tensor result = backend().rand(shape, type);
  trace("rand", {}, {}, {{"result", result}});
  return result;
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
    return result;                                                     \
  }                                                                    \
  Tensor TracerBackendBase::full(                                      \
      const Shape& dims, TYPE value, const dtype type) {               \
    Tensor result = backend().full(dims, value, type);                 \
    trace(                                                             \
        "full",                                                        \
        {{"dims", dims}, {"value", value}, {"type", type}},            \
        {},                                                            \
        {{"result", result}});                                         \
    return result;                                                     \
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
  return result;
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
  return result;
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
  return result;
}

/************************ Shaping and Indexing *************************/
Tensor TracerBackendBase::reshape(const Tensor& tensor, const Shape& shape) {
  Tensor result = tensor.backend().reshape(tensor, shape);
  trace(
      "reshape",
      {{"shape", shape}},
      {{"tensor", tensor}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::transpose(
    const Tensor& tensor,
    const Shape& axes /* = {} */) {
  Tensor result = tensor.backend().transpose(tensor, axes);
  trace(
      "transpose",
      {{"axes", axes}},
      {{"tensor", tensor}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::tile(const Tensor& tensor, const Shape& shape) {
  Tensor result = tensor.backend().tile(tensor, shape);
  trace("tile", {{"shape", shape}}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::concatenate(
    const std::vector<Tensor>& tensors,
    const unsigned axis) {
  Tensor result = tensors.front().backend().concatenate(tensors, axis);
  trace(
      "concatenate",
      {{"axis", axis}},
      {{"tensors", tensors}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::nonzero(const Tensor& tensor) {
  Tensor result = tensor.backend().nonzero(tensor);
  trace("nonzero", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::pad(
    const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type) {
  Tensor result = input.backend().pad(input, padWidths, type);
  trace(
      "pad",
      {{"padWidths", padWidths}, {"type", type}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

/************************** Unary Operators ***************************/

Tensor TracerBackendBase::exp(const Tensor& tensor) {
  Tensor result = tensor.backend().exp(tensor);
  trace("exp", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::log(const Tensor& tensor) {
  Tensor result = tensor.backend().log(tensor);
  trace("log", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::negative(const Tensor& tensor) {
  Tensor result = tensor.backend().negative(tensor);
  trace("negative", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::logicalNot(const Tensor& tensor) {
  Tensor result = tensor.backend().logicalNot(tensor);
  trace("logicalNot", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::log1p(const Tensor& tensor) {
  Tensor result = tensor.backend().log1p(tensor);
  trace("log1p", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::sin(const Tensor& tensor) {
  Tensor result = tensor.backend().sin(tensor);
  trace("sin", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::cos(const Tensor& tensor) {
  Tensor result = tensor.backend().cos(tensor);
  trace("cos", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::sqrt(const Tensor& tensor) {
  Tensor result = tensor.backend().sqrt(tensor);
  trace("sqrt", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::tanh(const Tensor& tensor) {
  Tensor result = tensor.backend().tanh(tensor);
  trace("tanh", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::floor(const Tensor& tensor) {
  Tensor result = tensor.backend().floor(tensor);
  trace("floor", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::ceil(const Tensor& tensor) {
  Tensor result = tensor.backend().ceil(tensor);
  trace("ceil", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::rint(const Tensor& tensor) {
  Tensor result = tensor.backend().rint(tensor);
  trace("rint", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::absolute(const Tensor& tensor) {
  Tensor result = tensor.backend().absolute(tensor);
  trace("absolute", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::sigmoid(const Tensor& tensor) {
  Tensor result = tensor.backend().sigmoid(tensor);
  trace("sigmoid", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::erf(const Tensor& tensor) {
  Tensor result = tensor.backend().erf(tensor);
  trace("erf", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::flip(const Tensor& tensor, const unsigned dim) {
  Tensor result = tensor.backend().flip(tensor, dim);
  trace("flip", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::clip(
    const Tensor& tensor,
    const Tensor& low,
    const Tensor& high) {
  Tensor result = tensor.backend().clip(tensor, low, high);
  trace(
      "clip",
      {{"low", low}, {"high", high}},
      {{"tensor", tensor}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::clip(
    const Tensor& tensor,
    const double& low,
    const double& high) {
  Tensor result = tensor.backend().clip(tensor, low, high);
  trace(
      "clip",
      {{"low", low}, {"high", high}},
      {{"tensor", tensor}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::roll(
    const Tensor& tensor,
    const int shift,
    const unsigned axis) {
  Tensor result = tensor.backend().roll(tensor, shift, axis);
  trace(
      "roll",
      {{"shift", shift}, {"axis", axis}},
      {{"tensor", tensor}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::isnan(const Tensor& tensor) {
  Tensor result = tensor.backend().isnan(tensor);
  trace("isnan", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::isinf(const Tensor& tensor) {
  Tensor result = tensor.backend().isinf(tensor);
  trace("isinf", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::sign(const Tensor& tensor) {
  Tensor result = tensor.backend().sign(tensor);
  trace("sign", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::tril(const Tensor& tensor) {
  Tensor result = tensor.backend().tril(tensor);
  trace("tril", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::triu(const Tensor& tensor) {
  Tensor result = tensor.backend().triu(tensor);
  trace("triu", {}, {{"tensor", tensor}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::where(
    const Tensor& condition,
    const Tensor& x,
    const Tensor& y) {
  Tensor result = condition.backend().where(condition, x, y);
  trace(
      "where",
      {},
      {{"condition", condition}, {"x", x}, {"y", y}},
      {{"result", result}});
  return result;
}

void TracerBackendBase::topk(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned k,
    const Dim axis,
    const SortMode sortMode /* = SortMode::Descending */) {
  trace(
      "topk",
      {{"input", input}, {"k", k}, {"axis", axis}, {"sortMode", sortMode}},
      {{"values", values}, {"indices", indices}},
      {});
  input.backend().topk(values, indices, input, k, axis, sortMode);
}

Tensor TracerBackendBase::sort(
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode) {
  Tensor result = input.backend().sort(input, axis, sortMode);
  trace(
      "sort",
      {{"axis", axis}, {"sortMode", sortMode}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

void TracerBackendBase::sort(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode /* = SortMode::Descending */) {
  trace(
      "sort",
      {{"input", input}, {"axis", axis}, {"sortMode", sortMode}},
      {{"values", values}, {"indices", indices}},
      {});
  values.backend().sort(values, indices, input, axis, sortMode);
}

Tensor TracerBackendBase::argsort(
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode) {
  Tensor result = input.backend().argsort(input, axis, sortMode);
  trace(
      "argsort",
      {{"axis", axis}, {"sortMode", sortMode}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

/************************** Binary Operators ***************************/
#define FL_TRACER_BINARY_OP_TYPE_DEF(FUNC, OP, TYPE)                      \
  Tensor TracerBackendBase::FUNC(const Tensor& lhs, TYPE rhs) {           \
    Tensor result = lhs OP rhs;                                           \
    trace(#FUNC, {}, {{"lhs", lhs}, {"rhs", rhs}}, {{"result", result}}); \
    return result;                                                        \
  }                                                                       \
  Tensor TracerBackendBase::FUNC(TYPE lhs, const Tensor& rhs) {           \
    Tensor result = lhs OP rhs;                                           \
    trace(#FUNC, {}, {{"lhs", lhs}, {"rhs", rhs}}, {{"result", result}}); \
    return result;                                                        \
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
#define FL_TRACER_BINARY_OP_DEF(OP, FUNC)                                 \
  Tensor TracerBackendBase::FUNC(const Tensor& lhs, const Tensor& rhs) {  \
    Tensor result = lhs OP rhs;                                           \
    trace(#FUNC, {}, {{"lhs", lhs}, {"rhs", rhs}}, {{"result", result}}); \
    return result;                                                        \
  }                                                                       \
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

Tensor TracerBackendBase::minimum(const Tensor& lhs, const Tensor& rhs) {
  Tensor result = lhs.backend().minimum(lhs, rhs);
  trace("minumum", {}, {{"lhs", lhs}, {"rhs", rhs}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::maximum(const Tensor& lhs, const Tensor& rhs) {
  Tensor result = lhs.backend().maximum(lhs, rhs);
  trace("maximum", {}, {{"lhs", lhs}, {"rhs", rhs}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::power(const Tensor& lhs, const Tensor& rhs) {
  Tensor result = lhs.backend().power(lhs, rhs);
  trace("power", {}, {{"lhs", lhs}, {"rhs", rhs}}, {{"result", result}});
  return result;
}

/************************** BLAS ***************************/

Tensor TracerBackendBase::matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp) {
  Tensor result = lhs.backend().matmul(lhs, rhs, lhsProp, rhsProp);
  trace(
      "matmul",
      {{"lhsProp", lhsProp}, {"rhsProp", rhsProp}},
      {{"lhs", lhs}, {"rhs", rhs}},
      {{"result", result}});
  return result;
}
/************************** Reductions ***************************/

Tensor TracerBackendBase::amin(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor result = input.backend().amin(input, axes, keepDims);
  trace(
      "amin",
      {{"axes", axes}, {"keepDims", keepDims}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::amax(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor result = input.backend().amax(input, axes, keepDims);
  trace(
      "amax",
      {{"axes", axes}, {"keepDims", keepDims}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

void TracerBackendBase::min(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  trace(
      "min",
      {{"axis", axis}, {"keepDims", keepDims}},
      {{"values", values}, {"indices", indices}, {"input", input}},
      {});
  input.backend().min(values, indices, input, axis, keepDims);
}

void TracerBackendBase::max(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims /* = false */) {
  trace(
      "max",
      {{"axis", axis}, {"keepDims", keepDims}},
      {{"values", values}, {"indices", indices}, {"input", input}},
      {});
  input.backend().max(values, indices, input, axis, keepDims);
}

Tensor TracerBackendBase::sum(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor result = input.backend().sum(input, axes, keepDims);
  trace(
      "sum",
      {{"axes", axes}, {"keepDims", keepDims}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::cumsum(const Tensor& input, const unsigned axis) {
  Tensor result = input.backend().cumsum(input, axis);
  trace("cumsum", {{"axis", axis}}, {{"input", input}}, {{"result", result}});
  return result;
}

Tensor TracerBackendBase::argmax(
    const Tensor& input,
    const unsigned axis,
    const bool keepDims /* = false */) {
  Tensor result = input.backend().argmax(input, axis, keepDims);
  trace(
      "argmax",
      {{"axis", axis}, {"keepDims", keepDims}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::argmin(
    const Tensor& input,
    const unsigned axis,
    const bool keepDims /* = false */) {
  Tensor result = input.backend().argmin(input, axis, keepDims);
  trace(
      "argmin",
      {{"axis", axis}, {"keepDims", keepDims}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::mean(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor result = input.backend().mean(input, axes, keepDims);
  trace(
      "mean",
      {{"axes", axes}, {"keepDims", keepDims}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::median(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor result = input.backend().median(input, axes, keepDims);
  trace(
      "median",
      {{"axes", axes}, {"keepDims", keepDims}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::var(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool bias,
    const bool keepDims /* = false */) {
  Tensor result = input.backend().var(input, axes, bias, keepDims);
  trace(
      "var",
      {{"axes", axes}, {"bias", bias}, {"keepDims", keepDims}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::std(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor result = input.backend().std(input, axes, keepDims);
  trace(
      "std",
      {{"axes", axes}, {"keepDims", keepDims}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::norm(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    double p /* = 2 */,
    const bool keepDims /* = false */) {
  Tensor result = input.backend().norm(input, axes, p, keepDims);
  trace(
      "norm",
      {{"axes", axes}, {"p", p}, {"keepDims", keepDims}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::countNonzero(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor result = input.backend().countNonzero(input, axes, keepDims);
  trace(
      "countNonzero",
      {{"axes", axes}, {"keepDims", keepDims}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::any(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor result = input.backend().any(input, axes, keepDims);
  trace(
      "any",
      {{"axes", axes}, {"keepDims", keepDims}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

Tensor TracerBackendBase::all(
    const Tensor& input,
    const std::vector<int>& axes /* = {} */,
    const bool keepDims /* = false */) {
  Tensor result = input.backend().all(input, axes, keepDims);
  trace(
      "all",
      {{"axes", axes}, {"keepDims", keepDims}},
      {{"input", input}},
      {{"result", result}});
  return result;
}

void TracerBackendBase::print(const Tensor& tensor) {
  tensor.backend().print(tensor);
}

} // namespace fl
