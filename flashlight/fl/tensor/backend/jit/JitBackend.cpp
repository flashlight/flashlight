/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/JitBackend.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ShapeInference.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

namespace {

std::vector<NodePtr> tensorsToNodes(const std::vector<Tensor>& tensors) {
  std::vector<NodePtr> nodes;
  // JitTensor copy is ~free, but whatever
  for (const auto& tensor : tensors) {
    nodes.push_back(toJitTensorBase(tensor).node());
  }
  return nodes;
}

template <typename... T>
std::vector<NodePtr> tensorsToNodes(const T&... tensors) {
  std::vector<NodePtr> nodes;
  // JitTensor copy is ~free, but whatever
  for (const auto& tensor : {&tensors...}) {
    nodes.push_back(toJitTensorBase(*tensor).node());
  }
  return nodes;
}

template <>
std::vector<NodePtr> tensorsToNodes() {
  return {};
}

const Tensor& materialize(Tensor tensor) {
  auto& jitTensor = toJitTensorBase(tensor);
  jitTensor.eval();
  return jitTensor.node()->getResult().value();
}

} // namespace

JitBackend::JitBackend(
    TensorBackend& wrappedBackend,
    std::function<Tensor(NodePtr)> jitTensorCreator)
    : wrappedBackend_(wrappedBackend),
      jitTensorCreator_(jitTensorCreator),
      evaluator_(wrappedBackend),
      optimizer_(wrappedBackend) {}

TensorBackendType JitBackend::backendType() const {
  return TensorBackendType::Jit;
}

Evaluator& JitBackend::evaluator() {
  return evaluator_;
}

Optimizer& JitBackend::optimizer() {
  return optimizer_;
}

TensorBackend& JitBackend::wrappedBackend() {
  return wrappedBackend_;
}

/* -------------------------- Compute Functions -------------------------- */

void JitBackend::eval(const Tensor& tensor) {
  auto& jitTensor = toJitTensorBase(tensor);
  assert(&jitTensor.backend() == this);
  jitTensor.eval();
  wrappedBackend_.eval(jitTensor.node()->getResult().value()); // "deep" eval
}

bool JitBackend::supportsDataType(const fl::dtype& dtype) const {
  return wrappedBackend_.supportsDataType(dtype);
}

void JitBackend::getMemMgrInfo(
    const char* msg,
    const int deviceId,
    std::ostream* ostream) {
  return wrappedBackend_.getMemMgrInfo(msg, deviceId, ostream);
}

void JitBackend::setMemMgrLogStream(std::ostream* stream) {
  return wrappedBackend_.setMemMgrLogStream(stream);
}

void JitBackend::setMemMgrLoggingEnabled(const bool enabled) {
  return wrappedBackend_.setMemMgrLoggingEnabled(enabled);
}

void JitBackend::setMemMgrFlushInterval(const size_t interval) {
  return wrappedBackend_.setMemMgrFlushInterval(interval);
}

/* -------------------------- Rand Functions -------------------------- */

void JitBackend::setSeed(const int seed) {
  wrappedBackend_.setSeed(seed);
}

Tensor JitBackend::randn(const Shape& shape, dtype type) {
  return jitTensorCreator_(CustomNode::create(
      "randn",
      tensorsToNodes(),
      Shape(shape),
      [this, shape, type](const std::vector<const Tensor*>& /* inputs */) {
        return wrappedBackend_.randn(shape, type);
      }));
}

Tensor JitBackend::rand(const Shape& shape, dtype type) {
  return jitTensorCreator_(CustomNode::create(
      "rand",
      tensorsToNodes(),
      Shape(shape),
      [this, shape, type](const std::vector<const Tensor*>& /* inputs */) {
        return wrappedBackend_.rand(shape, type);
      }));
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
#define FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(TYPE)                      \
  Tensor JitBackend::fromScalar(TYPE value, const dtype type) {               \
    return full(Shape{}, value, type);                                        \
  }                                                                           \
  Tensor JitBackend::full(const Shape& shape, TYPE value, const dtype type) { \
    return fullWithType(shape, value, type);                                  \
  }
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const double&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const float&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const int&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const unsigned&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const char&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const unsigned char&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const long&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const unsigned long&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const long long&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const unsigned long long&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const bool&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const short&);
FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(const unsigned short&);

template <typename T>
Tensor JitBackend::fullWithType(const Shape& shape, T value, dtype type) {
  return jitTensorCreator_(ScalarNode::create(shape, type, value));
}

Tensor JitBackend::identity(const Dim dim, const dtype type) {
  return jitTensorCreator_(
      ValueNode::create(wrappedBackend_.identity(dim, type)));
}

Tensor
JitBackend::arange(const Shape& shape, const Dim seqDim, const dtype type) {
  return jitTensorCreator_(CustomNode::create(
      "arange",
      tensorsToNodes(),
      Shape(shape),
      [this, shape, seqDim, type](
          const std::vector<const Tensor*>& /* inputs */) {
        return wrappedBackend_.arange(shape, seqDim, type);
      }));
}

Tensor
JitBackend::iota(const Shape& dims, const Shape& tileDims, const dtype type) {
  return jitTensorCreator_(
      ValueNode::create(wrappedBackend_.iota(dims, tileDims, type)));
}

/************************ Shaping and Indexing *************************/
Tensor JitBackend::reshape(const Tensor& tensor, const Shape& shape) {
  if (tensor.shape().elements() != shape.elements()) {
    std::ostringstream oss;
    oss << "[JitBackend::reshape] Cannot reshape from " << tensor.shape()
        << " to " << shape;
    throw std::invalid_argument(oss.str());
  }
  return jitTensorCreator_(CustomNode::create(
      "reshape",
      tensorsToNodes(tensor),
      Shape(shape),
      [this, shape](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.reshape(*inputs.at(0), shape);
      }));
}

Tensor JitBackend::transpose(const Tensor& tensor, const Shape& axes = {}) {
  return jitTensorCreator_(CustomNode::create(
      "transpose",
      tensorsToNodes(tensor),
      inferTransposeOutputShape(tensor.shape(), axes),
      [this, axes](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.transpose(*inputs.at(0), axes);
      }));
}

Tensor JitBackend::tile(const Tensor& tensor, const Shape& tileDims) {
  return jitTensorCreator_(CustomNode::create(
      "tile",
      tensorsToNodes(tensor),
      inferTileOutputShape(tensor.shape(), tileDims),
      [this, tileDims](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.tile(*inputs[0], tileDims);
      }));
}

Tensor JitBackend::concatenate(
    const std::vector<Tensor>& tensors,
    const unsigned axisToConcat) {
  return jitTensorCreator_(CustomNode::create(
      "concatenate",
      tensorsToNodes(tensors),
      inferConcatenateOutputShape(tensors, axisToConcat),
      [this, axisToConcat](const std::vector<const Tensor*>& inputs) {
        // TODO use shallowcopy here
        std::vector<Tensor> inputTensors;
        for (const auto* inputPtr : inputs) {
          inputTensors.emplace_back(inputPtr->copy());
        }
        return wrappedBackend_.concatenate(inputTensors, axisToConcat);
      }));
}

Tensor JitBackend::nonzero(const Tensor& tensor) {
  // NOTE must materialize since we can't infer output shape
  const auto& tensorResult = materialize(tensor);
  return jitTensorCreator_(
      ValueNode::create(wrappedBackend_.nonzero(tensorResult)));
}

Tensor JitBackend::pad(
    const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type) {
  return jitTensorCreator_(CustomNode::create(
      "pad",
      tensorsToNodes(input),
      inferPadOutputShape(input.shape(), padWidths),
      [this, padWidths, type](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.pad(*inputs.at(0), padWidths, type);
      }));
}

/************************** Unary Operators ***************************/

#define FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(OP)             \
  {                                                        \
    return jitTensorCreator_(CustomNode::create(           \
        #OP,                                               \
        tensorsToNodes(tensor),                            \
        Shape(tensor.shape()),                             \
        [this](const std::vector<const Tensor*>& inputs) { \
          return wrappedBackend_.OP(*inputs.at(0));        \
        }));                                               \
  }

Tensor JitBackend::exp(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(exp);
}

Tensor JitBackend::log(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(log);
}

Tensor JitBackend::negative(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(negative);
}

Tensor JitBackend::logicalNot(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(logicalNot);
}

Tensor JitBackend::log1p(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(log1p);
}

Tensor JitBackend::sin(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(sin);
}

Tensor JitBackend::cos(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(cos);
}

Tensor JitBackend::sqrt(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(sqrt);
}

Tensor JitBackend::tanh(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(tanh);
}

Tensor JitBackend::floor(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(floor);
}

Tensor JitBackend::ceil(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(ceil);
}

Tensor JitBackend::rint(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(rint);
}

Tensor JitBackend::absolute(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(absolute);
}

Tensor JitBackend::sigmoid(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(sigmoid);
}

Tensor JitBackend::erf(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(erf);
}

Tensor JitBackend::flip(const Tensor& tensor, const unsigned dim) {
  return jitTensorCreator_(CustomNode::create(
      "flip",
      tensorsToNodes(tensor),
      Shape(tensor.shape()),
      [this, dim](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.flip(*inputs.at(0), dim);
      }));
}

Tensor
JitBackend::clip(const Tensor& tensor, const Tensor& low, const Tensor& high) {
  return jitTensorCreator_(CustomNode::create(
      "clip",
      tensorsToNodes(tensor, low, high),
      Shape(tensor.shape()),
      [this](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.clip(
            *inputs.at(0), *inputs.at(1), *inputs.at(2));
      }));
}

Tensor
JitBackend::roll(const Tensor& tensor, const int shift, const unsigned axis) {
  return jitTensorCreator_(CustomNode::create(
      "roll",
      tensorsToNodes(tensor),
      Shape(tensor.shape()),
      [this, shift, axis](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.roll(*inputs.at(0), shift, axis);
      }));
}

Tensor JitBackend::isnan(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(isnan);
}

Tensor JitBackend::isinf(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(isinf);
}

Tensor JitBackend::sign(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(sign);
}

Tensor JitBackend::tril(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(tril);
}

Tensor JitBackend::triu(const Tensor& tensor) {
  FL_JIT_BACKEND_UNARY_FALLBACK_IMPL(triu);
}
#undef FL_JIT_BACKEND_UNARY_FALLBACK_IMPL

Tensor
JitBackend::where(const Tensor& condition, const Tensor& x, const Tensor& y) {
  return jitTensorCreator_(CustomNode::create(
      "where",
      tensorsToNodes(condition, x, y),
      Shape(condition.shape()),
      [this](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.where(
            *inputs.at(0), *inputs.at(1), *inputs.at(2));
      }));
}

void JitBackend::topk(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned k,
    const Dim axis,
    const SortMode sortMode) {
  const auto& inputResult = materialize(input);
  auto valuesResult = this->full({1}, 0, dtype::s32);
  auto indicesResult = this->full({1}, 0, dtype::s32);
  wrappedBackend_.topk(
      valuesResult, indicesResult, inputResult, k, axis, sortMode);
  values = jitTensorCreator_(ValueNode::create(std::move(valuesResult)));
  indices = jitTensorCreator_(ValueNode::create(std::move(indicesResult)));
}

Tensor
JitBackend::sort(const Tensor& input, const Dim axis, const SortMode sortMode) {
  return jitTensorCreator_(CustomNode::create(
      "sort",
      tensorsToNodes(input),
      Shape(input.shape()),
      [this, axis, sortMode](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.sort(*inputs.at(0), axis, sortMode);
      }));
}

void JitBackend::sort(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode) {
  const auto& inputResult = materialize(input);
  auto valuesResult = this->full({1}, 0, dtype::s32);
  auto indicesResult = this->full({1}, 0, dtype::s32);
  wrappedBackend_.sort(
      valuesResult, indicesResult, inputResult, axis, sortMode);
  values = jitTensorCreator_(ValueNode::create(std::move(valuesResult)));
  indices = jitTensorCreator_(ValueNode::create(std::move(indicesResult)));
}

Tensor JitBackend::argsort(
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode) {
  return jitTensorCreator_(CustomNode::create(
      "argsort",
      tensorsToNodes(input),
      Shape(input.shape()),
      [this, axis, sortMode](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.argsort(*inputs.at(0), axis, sortMode);
      }));
}

/************************** Binary Operators ***************************/
Tensor JitBackend::createBinopJitTensor(
    const Tensor& lhs,
    const Tensor& rhs,
    BinaryOp op) {
  const auto lhsNode = toJitTensorBase(lhs).node();
  const auto rhsNode = toJitTensorBase(rhs).node();
  return jitTensorCreator_(BinaryNode::create(lhsNode, rhsNode, op));
}

// TODO remove once onednn backend supports horizontal broadcast
template <typename T>
Tensor JitBackend::createScalarTensor(unsigned ndim, T val) {
  Shape literalShape(std::vector<Dim>(ndim, 1));
  auto type = dtype_traits<T>::fl_type;
  return full(literalShape, val, type);
}

#define FL_JIT_BINARY_OP_TYPE_DEF(FUNC, TYPE)          \
  Tensor JitBackend::FUNC(const Tensor& a, TYPE rhs) { \
    return FUNC(a, createScalarTensor(a.ndim(), rhs)); \
  }                                                    \
  Tensor JitBackend::FUNC(TYPE lhs, const Tensor& a) { \
    return FUNC(createScalarTensor(a.ndim(), lhs), a); \
  }

#define FL_JIT_BINARY_OP_LITERALS_DEF(FUNC)                   \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const bool&);               \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const int&);                \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const unsigned&);           \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const char&);               \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const unsigned char&);      \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const long&);               \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const unsigned long&);      \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const long long&);          \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const unsigned long long&); \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const double&);             \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const float&);              \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const short&);              \
  FL_JIT_BINARY_OP_TYPE_DEF(FUNC, const unsigned short&);

#define FL_JIT_BINARY_OP_TENSOR_DEF(FUNC, BINOP)                  \
  Tensor JitBackend::FUNC(const Tensor& lhs, const Tensor& rhs) { \
    return createBinopJitTensor(lhs, rhs, BINOP);                 \
  }                                                               \
  FL_JIT_BINARY_OP_LITERALS_DEF(FUNC);

FL_JIT_BINARY_OP_TENSOR_DEF(add, BinaryOp::Add);
FL_JIT_BINARY_OP_TENSOR_DEF(sub, BinaryOp::Sub);
FL_JIT_BINARY_OP_TENSOR_DEF(mul, BinaryOp::Mul);
FL_JIT_BINARY_OP_TENSOR_DEF(div, BinaryOp::Div);
FL_JIT_BINARY_OP_TENSOR_DEF(eq, BinaryOp::Eq);
FL_JIT_BINARY_OP_TENSOR_DEF(neq, BinaryOp::Neq);
FL_JIT_BINARY_OP_TENSOR_DEF(lessThan, BinaryOp::Lt);
FL_JIT_BINARY_OP_TENSOR_DEF(lessThanEqual, BinaryOp::Lte);
FL_JIT_BINARY_OP_TENSOR_DEF(greaterThan, BinaryOp::Gt);
FL_JIT_BINARY_OP_TENSOR_DEF(greaterThanEqual, BinaryOp::Gte);
FL_JIT_BINARY_OP_TENSOR_DEF(logicalOr, BinaryOp::Or);
FL_JIT_BINARY_OP_TENSOR_DEF(logicalAnd, BinaryOp::And);
FL_JIT_BINARY_OP_TENSOR_DEF(mod, BinaryOp::Mod);
FL_JIT_BINARY_OP_TENSOR_DEF(bitwiseAnd, BinaryOp::BitAnd);
FL_JIT_BINARY_OP_TENSOR_DEF(bitwiseOr, BinaryOp::BitOr);
FL_JIT_BINARY_OP_TENSOR_DEF(bitwiseXor, BinaryOp::BitXor);
FL_JIT_BINARY_OP_TENSOR_DEF(lShift, BinaryOp::Shl);
FL_JIT_BINARY_OP_TENSOR_DEF(rShift, BinaryOp::Shr);
#undef FL_JIT_BINARY_OP_TYPE_DEF
#undef FL_JIT_BINARY_OP_LITERALS_DEF
#undef FL_JIT_BINARY_OP_TENSOR_DEF

Tensor JitBackend::minimum(const Tensor& lhs, const Tensor& rhs) {
  return createBinopJitTensor(lhs, rhs, BinaryOp::Min);
}

Tensor JitBackend::maximum(const Tensor& lhs, const Tensor& rhs) {
  return createBinopJitTensor(lhs, rhs, BinaryOp::Max);
}

Tensor JitBackend::power(const Tensor& lhs, const Tensor& rhs) {
  return createBinopJitTensor(lhs, rhs, BinaryOp::Pow);
}

/************************** BLAS ***************************/

Tensor JitBackend::matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp) {
  return jitTensorCreator_(CustomNode::create(
      "matmul",
      tensorsToNodes(lhs, rhs),
      inferMatmulOutputShape(lhs.shape(), rhs.shape(), lhsProp, rhsProp),
      [this, lhsProp, rhsProp](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.matmul(
            *inputs.at(0), *inputs.at(1), lhsProp, rhsProp);
      }));
}

/************************** Reductions ***************************/

#define FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(OP)                         \
  {                                                                        \
    return jitTensorCreator_(CustomNode::create(                           \
        #OP,                                                               \
        tensorsToNodes(input),                                             \
        inferReductionOutputShape(input.shape(), axes, keepDims),          \
        [this, axes, keepDims](const std::vector<const Tensor*>& inputs) { \
          return wrappedBackend_.OP(*inputs.at(0), axes, keepDims);        \
        }));                                                               \
  }

Tensor JitBackend::amin(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(amin);
}

Tensor JitBackend::amax(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(amax);
}

void JitBackend::min(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  // TODO generalize IR to support nodes with 2 outputs? Need to investigate its
  // benefit on optimization
  const auto& inputResult = materialize(input);
  auto valuesResult = this->wrappedBackend_.full({}, 0, dtype::s32);
  auto indicesResult = this->wrappedBackend_.full({}, 0, dtype::s32);
  wrappedBackend_.min(valuesResult, indicesResult, inputResult, axis, keepDims);
  values = jitTensorCreator_(ValueNode::create(std::move(valuesResult)));
  indices = jitTensorCreator_(ValueNode::create(std::move(indicesResult)));
}

void JitBackend::max(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  const auto& inputResult = materialize(input);
  auto valuesResult = this->wrappedBackend_.full({}, 0, dtype::s32);
  auto indicesResult = this->wrappedBackend_.full({}, 0, dtype::s32);
  wrappedBackend_.max(valuesResult, indicesResult, inputResult, axis, keepDims);
  values = jitTensorCreator_(ValueNode::create(std::move(valuesResult)));
  indices = jitTensorCreator_(ValueNode::create(std::move(indicesResult)));
}

Tensor JitBackend::sum(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(sum);
}

Tensor JitBackend::cumsum(const Tensor& input, const unsigned axis) {
  return jitTensorCreator_(CustomNode::create(
      "cumsum",
      tensorsToNodes(input),
      Shape(input.shape()),
      [this, axis](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.cumsum(*inputs.at(0), axis);
      }));
}

Tensor JitBackend::argmax(
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  return jitTensorCreator_(CustomNode::create(
      "argmax",
      tensorsToNodes(input),
      inferReductionOutputShape(
          input.shape(), {static_cast<int>(axis)}, keepDims),
      [this, axis, keepDims](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.argmax(*inputs.at(0), axis, keepDims);
      }));
}

Tensor JitBackend::argmin(
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  return jitTensorCreator_(CustomNode::create(
      "argmax",
      tensorsToNodes(input),
      inferReductionOutputShape(
          input.shape(), {static_cast<int>(axis)}, keepDims),
      [this, axis, keepDims](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.argmin(*inputs.at(0), axis, keepDims);
      }));
}

Tensor JitBackend::mean(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(mean);
}

Tensor JitBackend::median(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(median);
}

Tensor JitBackend::var(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool bias,
    const bool keepDims) {
  return jitTensorCreator_(CustomNode::create(
      "var",
      tensorsToNodes(input),
      inferReductionOutputShape(input.shape(), axes, keepDims),
      [this, axes, bias, keepDims](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.var(*inputs.at(0), axes, bias, keepDims);
      }));
}

Tensor JitBackend::std(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(std);
}

Tensor JitBackend::norm(
    const Tensor& input,
    const std::vector<int>& axes,
    double p /* = 2 */,
    const bool keepDims) {
  return jitTensorCreator_(CustomNode::create(
      "norm",
      tensorsToNodes(input),
      inferReductionOutputShape(input.shape(), axes, keepDims),
      [this, axes, p, keepDims](const std::vector<const Tensor*>& inputs) {
        return wrappedBackend_.norm(*inputs.at(0), axes, p, keepDims);
      }));
}

Tensor JitBackend::countNonzero(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(countNonzero);
}

Tensor JitBackend::any(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(any);
}

Tensor JitBackend::all(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  FL_JIT_BACKEND_REDUCTION_FALLBACK_IMPL(all);
}

void JitBackend::print(const Tensor& tensor) {
  std::cout << "JitTensor" << std::endl
            << toJitTensorBase(const_cast<Tensor&>(tensor)).toString()
            << std::endl;
}

} // namespace fl
