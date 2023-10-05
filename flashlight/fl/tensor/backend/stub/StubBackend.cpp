/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/stub/StubBackend.h"

#include <stdexcept>

#include "flashlight/fl/tensor/TensorBase.h"

#define FL_STUB_BACKEND_UNIMPLEMENTED \
  throw std::invalid_argument(        \
      "StubBackend::" + std::string(__func__) + " - unimplemented.");

namespace fl {

StubBackend::StubBackend() {
  // Set up state
}

StubBackend& StubBackend::getInstance() {
  static StubBackend instance;
  return instance;
}

TensorBackendType StubBackend::backendType() const {
  // Implementers of a backend should create their own option in the
  // TensorBackendType enum and return it here.
  return TensorBackendType::Stub;
}

/* -------------------------- Compute Functions -------------------------- */

void StubBackend::eval(const Tensor& /* tensor */) {
  // Launch computation for a given tensor. Can be a noop for non-async
  // runtimes.
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

bool StubBackend::supportsDataType(const fl::dtype& /* dtype */) const {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::getMemMgrInfo(
    const char* /* msg */,
    const int /* deviceId */,
    std::ostream* /* ostream */) {
  // Can be a noop if no memory manager is implemented.
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::setMemMgrLogStream(std::ostream* /* stream */) {
  // Can be a noop if no memory manager is implemented.
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::setMemMgrLoggingEnabled(const bool /* enabled */) {
  // Can be a noop if no memory manager is implemented.
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::setMemMgrFlushInterval(const size_t /* interval */) {
  // Can be a noop if no memory manager is implemented.
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/* -------------------------- Rand Functions -------------------------- */

void StubBackend::setSeed(const int /* seed */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::randn(const Shape& /* shape */, dtype /* type */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::rand(const Shape& /* shape */, dtype /* type */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
#define FL_STUB_BACKEND_CREATE_FUN_LITERAL_DEF(TYPE)                           \
  Tensor StubBackend::fromScalar(TYPE /* value */, const dtype /* type */) {   \
    throw std::invalid_argument(                                               \
        "StubBackend::fromScalar - not implemented for type " +                \
        std::string(#TYPE));                                                   \
  }                                                                            \
  Tensor StubBackend::full(                                                    \
      const Shape& /* shape */, TYPE /* value */, const dtype /* type */) {    \
    throw std::invalid_argument(                                               \
        "StubBackend::full - not implemented for type " + std::string(#TYPE)); \
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

Tensor StubBackend::identity(const Dim /* dim */, const dtype /* type */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::arange(
    const Shape& /* shape */,
    const Dim /* seqDim */,
    const dtype /* type */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::iota(
    const Shape& /* dims */,
    const Shape& /* tileDims */,
    const dtype /* type */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/************************ Shaping and Indexing *************************/
Tensor StubBackend::reshape(
    const Tensor& /* tensor */,
    const Shape& /* shape */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::transpose(
    const Tensor& /* tensor */,
    const Shape& /* axes */ /* = {} */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::tile(const Tensor& /* tensor */, const Shape& /* shape */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::concatenate(
    const std::vector<Tensor>& /* tensors */,
    const unsigned /* axis */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::nonzero(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::pad(
    const Tensor& /* input */,
    const std::vector<std::pair<int, int>>& /* padWidths */,
    const PadType /* type */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/************************** Unary Operators ***************************/

Tensor StubBackend::exp(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::log(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::negative(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::logicalNot(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::log1p(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::sin(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::cos(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::sqrt(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::tanh(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::floor(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::ceil(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::rint(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::absolute(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::sigmoid(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::erf(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::flip(const Tensor& /* tensor */, const unsigned /* dim */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::clip(
    const Tensor& /* tensor */,
    const Tensor& /* low */,
    const Tensor& /* high */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::roll(
    const Tensor& /* tensor */,
    const int /* shift */,
    const unsigned /* axis */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::isnan(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::isinf(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::sign(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::tril(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::triu(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::where(
    const Tensor& /* condition */,
    const Tensor& /* x */,
    const Tensor& /* y */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::topk(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* k */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::sort(
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::sort(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::argsort(
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/************************** Binary Operators ***************************/
#define FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, TYPE)                            \
  Tensor StubBackend::FUNC(const Tensor& /* a */, TYPE /* rhs */) {         \
    throw std::runtime_error(                                               \
        "StubBackend::" + std::string(#FUNC) + " unimplemented for type " + \
        std::string(#TYPE));                                                \
  }                                                                         \
  Tensor StubBackend::FUNC(TYPE /* lhs */, const Tensor& /* a */) {         \
    throw std::runtime_error(                                               \
        "StubBackend::" + std::string(#FUNC) + " unimplemented for type " + \
        std::string(#TYPE));                                                \
  }

#define FL_AF_BINARY_OP_LITERALS_DEF(FUNC, OP)                   \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const bool&);               \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const int&);                \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned&);           \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const char&);               \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned char&);      \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const long&);               \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned long&);      \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const long long&);          \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned long long&); \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const double&);             \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const float&);              \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const short&);              \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned short&);

// Operations on fl::Tensor call the respective operator overloads that are
// already defined on af::arrays
#define FL_AF_BINARY_OP_DEF(OP, FUNC)                                          \
  Tensor StubBackend::FUNC(const Tensor& /* lhs */, const Tensor& /* rhs */) { \
    throw std::runtime_error(                                                  \
        "StubBackend::" + std::string(#FUNC) +                                 \
        " unimplemented for two-Tensor inputs.");                              \
  }                                                                            \
  FL_AF_BINARY_OP_LITERALS_DEF(FUNC, OP);

// Definitions
// Since ArrayFire implements operator overloads, map both fl::Tensor
// functions and fl::Tensor operator overloads back to the af::array
// overloads.
FL_AF_BINARY_OP_DEF(+, add);
FL_AF_BINARY_OP_DEF(-, sub);
FL_AF_BINARY_OP_DEF(*, mul);
FL_AF_BINARY_OP_DEF(/, div);
FL_AF_BINARY_OP_DEF(==, eq);
FL_AF_BINARY_OP_DEF(!=, neq);
FL_AF_BINARY_OP_DEF(<, lessThan);
FL_AF_BINARY_OP_DEF(<=, lessThanEqual);
FL_AF_BINARY_OP_DEF(>, greaterThan);
FL_AF_BINARY_OP_DEF(>=, greaterThanEqual);
FL_AF_BINARY_OP_DEF(||, logicalOr);
FL_AF_BINARY_OP_DEF(&&, logicalAnd);
FL_AF_BINARY_OP_DEF(%, mod);
FL_AF_BINARY_OP_DEF(&, bitwiseAnd);
FL_AF_BINARY_OP_DEF(|, bitwiseOr);
FL_AF_BINARY_OP_DEF(^, bitwiseXor);
FL_AF_BINARY_OP_DEF(<<, lShift);
FL_AF_BINARY_OP_DEF(>>, rShift);
#undef FL_AF_BINARY_OP_DEF
#undef FL_AF_BINARY_OP_TYPE_DEF
#undef FL_AF_BINARY_OP_LITERALS_DEF

Tensor StubBackend::minimum(const Tensor& /* lhs */, const Tensor& /* rhs */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::maximum(const Tensor& /* lhs */, const Tensor& /* rhs */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::power(const Tensor& /* lhs */, const Tensor& /* rhs */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/************************** BLAS ***************************/

Tensor StubBackend::matmul(
    const Tensor& /* lhs */,
    const Tensor& /* rhs */,
    MatrixProperty /* lhsProp */,
    MatrixProperty /* rhsProp */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

/************************** Reductions ***************************/

Tensor StubBackend::amin(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::amax(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::min(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::max(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::sum(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::cumsum(
    const Tensor& /* input */,
    const unsigned /* axis */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::argmax(
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::argmin(
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::mean(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::median(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::var(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* bias */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::std(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::norm(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    double /* p */ /* = 2 */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::countNonzero(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::any(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

Tensor StubBackend::all(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

void StubBackend::print(const Tensor& /* tensor */) {
  FL_STUB_BACKEND_UNIMPLEMENTED;
}

} // namespace fl
