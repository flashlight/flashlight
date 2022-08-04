/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/JitBackend.h"

#include <stdexcept>

#include "flashlight/fl/tensor/TensorBase.h"

#define FL_JIT_BACKEND_UNIMPLEMENTED \
  throw std::invalid_argument(       \
      "JitBackend::" + std::string(__func__) + " - unimplemented.");

namespace fl {

JitBackend::JitBackend(
    TensorBackend& wrappedBackend,
    std::function<Tensor(Node*)> jitTensorCreator)
    : wrappedBackend_(wrappedBackend), jitTensorCreator_(jitTensorCreator) {}

TensorBackendType JitBackend::backendType() const {
  return TensorBackendType::Jit;
}

/* -------------------------- Compute Functions -------------------------- */

void JitBackend::eval(const Tensor& /* tensor */) {
  // Launch computation for a given tensor. Can be a noop for non-async
  // runtimes.
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

bool JitBackend::supportsDataType(const fl::dtype& /* dtype */) const {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

void JitBackend::getMemMgrInfo(
    const char* /* msg */,
    const int /* deviceId */,
    std::ostream* /* ostream */) {
  // Can be a noop if no memory manager is implemented.
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

void JitBackend::setMemMgrLogStream(std::ostream* /* stream */) {
  // Can be a noop if no memory manager is implemented.
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

void JitBackend::setMemMgrLoggingEnabled(const bool /* enabled */) {
  // Can be a noop if no memory manager is implemented.
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

void JitBackend::setMemMgrFlushInterval(const size_t /* interval */) {
  // Can be a noop if no memory manager is implemented.
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

/* -------------------------- Rand Functions -------------------------- */

void JitBackend::setSeed(const int /* seed */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::randn(const Shape& /* shape */, dtype /* type */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::rand(const Shape& /* shape */, dtype /* type */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
#define FL_JIT_BACKEND_CREATE_FUN_LITERAL_DEF_STUB(TYPE)                      \
  Tensor JitBackend::fromScalar(TYPE /* value */, const dtype /* type */) {   \
    throw std::invalid_argument(                                              \
        "JitBackend::fromScalar - not implemented for type " +                \
        std::string(#TYPE));                                                  \
  }                                                                           \
  Tensor JitBackend::full(                                                    \
      const Shape& /* shape */, TYPE /* value */, const dtype /* type */) {   \
    throw std::invalid_argument(                                              \
        "JitBackend::full - not implemented for type " + std::string(#TYPE)); \
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

Tensor JitBackend::identity(const Dim /* dim */, const dtype /* type */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::arange(
    const Shape& /* shape */,
    const Dim /* seqDim */,
    const dtype /* type */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::iota(
    const Shape& /* dims */,
    const Shape& /* tileDims */,
    const dtype /* type */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

/************************ Shaping and Indexing *************************/
Tensor JitBackend::reshape(
    const Tensor& /* tensor */,
    const Shape& /* shape */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::transpose(
    const Tensor& /* tensor */,
    const Shape& /* axes */ /* = {} */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::tile(const Tensor& /* tensor */, const Shape& /* shape */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::concatenate(
    const std::vector<Tensor>& /* tensors */,
    const unsigned /* axis */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::nonzero(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::pad(
    const Tensor& /* input */,
    const std::vector<std::pair<int, int>>& /* padWidths */,
    const PadType /* type */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

/************************** Unary Operators ***************************/

Tensor JitBackend::exp(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::log(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::negative(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::logicalNot(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::log1p(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::sin(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::cos(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::sqrt(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::tanh(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::floor(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::ceil(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::rint(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::absolute(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::sigmoid(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::erf(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::flip(const Tensor& /* tensor */, const unsigned /* dim */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::clip(
    const Tensor& /* tensor */,
    const Tensor& /* low */,
    const Tensor& /* high */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::roll(
    const Tensor& /* tensor */,
    const int /* shift */,
    const unsigned /* axis */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::isnan(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::isinf(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::sign(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::tril(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::triu(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::where(
    const Tensor& /* condition */,
    const Tensor& /* x */,
    const Tensor& /* y */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

void JitBackend::topk(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* k */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::sort(
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

void JitBackend::sort(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::argsort(
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

/************************** Binary Operators ***************************/
#define FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, TYPE)                     \
  Tensor JitBackend::FUNC(const Tensor& /* a */, TYPE /* rhs */) {         \
    throw std::runtime_error(                                              \
        "JitBackend::" + std::string(#FUNC) + " unimplemented for type " + \
        std::string(#TYPE));                                               \
  }                                                                        \
  Tensor JitBackend::FUNC(TYPE /* lhs */, const Tensor& /* a */) {         \
    throw std::runtime_error(                                              \
        "JitBackend::" + std::string(#FUNC) + " unimplemented for type " + \
        std::string(#TYPE));                                               \
  }

#define FL_JIT_BINARY_OP_LITERALS_DEF_STUB(FUNC, OP)                   \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const bool&);               \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const int&);                \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const unsigned&);           \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const char&);               \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const unsigned char&);      \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const long&);               \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const unsigned long&);      \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const long long&);          \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const unsigned long long&); \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const double&);             \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const float&);              \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const short&);              \
  FL_JIT_BINARY_OP_TYPE_DEF_STUB(FUNC, OP, const unsigned short&);

// Operations on fl::Tensor call the respective operator overloads that are
// already defined on af::arrays
#define FL_JIT_BINARY_OP_DEF_STUB(OP, FUNC)                                   \
  Tensor JitBackend::FUNC(const Tensor& /* lhs */, const Tensor& /* rhs */) { \
    throw std::runtime_error(                                                 \
        "JitBackend::" + std::string(#FUNC) +                                 \
        " unimplemented for two-Tensor inputs.");                             \
  }                                                                           \
  FL_JIT_BINARY_OP_LITERALS_DEF_STUB(FUNC, OP);

// Definitions
// Since ArrayFire implements operator overloads, map both fl::Tensor
// functions and fl::Tensor operator overloads back to the af::array
// overloads.
FL_JIT_BINARY_OP_DEF_STUB(+, add);
FL_JIT_BINARY_OP_DEF_STUB(-, sub);
FL_JIT_BINARY_OP_DEF_STUB(*, mul);
FL_JIT_BINARY_OP_DEF_STUB(/, div);
FL_JIT_BINARY_OP_DEF_STUB(==, eq);
FL_JIT_BINARY_OP_DEF_STUB(!=, neq);
FL_JIT_BINARY_OP_DEF_STUB(<, lessThan);
FL_JIT_BINARY_OP_DEF_STUB(<=, lessThanEqual);
FL_JIT_BINARY_OP_DEF_STUB(>, greaterThan);
FL_JIT_BINARY_OP_DEF_STUB(>=, greaterThanEqual);
FL_JIT_BINARY_OP_DEF_STUB(||, logicalOr);
FL_JIT_BINARY_OP_DEF_STUB(&&, logicalAnd);
FL_JIT_BINARY_OP_DEF_STUB(%, mod);
FL_JIT_BINARY_OP_DEF_STUB(&, bitwiseAnd);
FL_JIT_BINARY_OP_DEF_STUB(|, bitwiseOr);
FL_JIT_BINARY_OP_DEF_STUB(^, bitwiseXor);
FL_JIT_BINARY_OP_DEF_STUB(<<, lShift);
FL_JIT_BINARY_OP_DEF_STUB(>>, rShift);
#undef FL_JIT_BINARY_OP_DEF
#undef FL_JIT_BINARY_OP_TYPE_DEF
#undef FL_JIT_BINARY_OP_LITERALS_DEF

Tensor JitBackend::minimum(const Tensor& /* lhs */, const Tensor& /* rhs */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::maximum(const Tensor& /* lhs */, const Tensor& /* rhs */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::power(const Tensor& /* lhs */, const Tensor& /* rhs */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

/************************** BLAS ***************************/

Tensor JitBackend::matmul(
    const Tensor& /* lhs */,
    const Tensor& /* rhs */,
    MatrixProperty /* lhsProp */,
    MatrixProperty /* rhsProp */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

/************************** Reductions ***************************/

Tensor JitBackend::amin(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::amax(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

void JitBackend::min(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

void JitBackend::max(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::sum(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::cumsum(
    const Tensor& /* input */,
    const unsigned /* axis */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::argmax(
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::argmin(
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::mean(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::median(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::var(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* bias */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::std(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::norm(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    double /* p */ /* = 2 */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::countNonzero(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::any(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

Tensor JitBackend::all(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

void JitBackend::print(const Tensor& /* tensor */) {
  FL_JIT_BACKEND_UNIMPLEMENTED;
}

} // namespace fl
