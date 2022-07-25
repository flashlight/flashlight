/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/onednn/OneDnnBackend.h"

#include <iostream>
#include <stdexcept>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"

#define FL_ONEDNN_BACKEND_UNIMPLEMENTED \
  throw std::invalid_argument(          \
      "OneDnnBackend::" + std::string(__func__) + " - unimplemented.");

namespace fl {

OneDnnBackend::OneDnnBackend() {
  // Set up state
}

OneDnnBackend& OneDnnBackend::getInstance() {
  static OneDnnBackend instance;
  return instance;
}

TensorBackendType OneDnnBackend::backendType() const {
  return TensorBackendType::OneDnn;
}

/* -------------------------- Compute Functions -------------------------- */

void OneDnnBackend::eval(const Tensor& /* tensor */) {
  // no-op since OneDNN computations are launched eagerly
}

bool OneDnnBackend::supportsDataType(const fl::dtype& /* dtype */) const {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

void OneDnnBackend::getMemMgrInfo(
    const char* /* msg */,
    const int /* deviceId */,
    std::ostream* /* ostream */) {
  throw std::runtime_error(
      "[OneDnnBackend] Currently no memory manager support");
}

void OneDnnBackend::setMemMgrLogStream(std::ostream* /* stream */) {
  throw std::runtime_error(
      "[OneDnnBackend] Currently no memory manager support");
}

void OneDnnBackend::setMemMgrLoggingEnabled(const bool /* enabled */) {
  throw std::runtime_error(
      "[OneDnnBackend] Currently no memory manager support");
}

void OneDnnBackend::setMemMgrFlushInterval(const size_t /* interval */) {
  throw std::runtime_error(
      "[OneDnnBackend] Currently no memory manager support");
}

/* -------------------------- Rand Functions -------------------------- */

void OneDnnBackend::setSeed(const int /* seed */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::randn(const Shape& /* shape */, dtype /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::rand(const Shape& /* shape */, dtype /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
#define FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(TYPE)                         \
  Tensor OneDnnBackend::fromScalar(TYPE /* value */, const dtype /* type */) { \
    throw std::invalid_argument(                                               \
        "OneDnnBackend::fromScalar - not implemented for type " +              \
        std::string(#TYPE));                                                   \
  }                                                                            \
  Tensor OneDnnBackend::full(                                                  \
      const Shape& /* shape */, TYPE /* value */, const dtype /* type */) {    \
    throw std::invalid_argument(                                               \
        "OneDnnBackend::full - not implemented for type " +                    \
        std::string(#TYPE));                                                   \
  }
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const double&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const float&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const int&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const char&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned char&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const long&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned long&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const long long&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned long long&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const bool&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const short&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned short&);

Tensor OneDnnBackend::identity(const Dim /* dim */, const dtype /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::arange(
    const Shape& /* shape */,
    const Dim /* seqDim */,
    const dtype /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::iota(
    const Shape& /* dims */,
    const Shape& /* tileDims */,
    const dtype /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

/************************ Shaping and Indexing *************************/
Tensor OneDnnBackend::reshape(
    const Tensor& /* tensor */,
    const Shape& /* shape */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::transpose(
    const Tensor& /* tensor */,
    const Shape& /* axes */ /* = {} */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::tile(
    const Tensor& /* tensor */,
    const Shape& /* shape */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::concatenate(
    const std::vector<Tensor>& /* tensors */,
    const unsigned /* axis */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::nonzero(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::pad(
    const Tensor& /* input */,
    const std::vector<std::pair<int, int>>& /* padWidths */,
    const PadType /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

/************************** Unary Operators ***************************/

Tensor OneDnnBackend::exp(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::log(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::negative(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::logicalNot(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::log1p(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::sin(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::cos(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::sqrt(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::tanh(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::floor(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::ceil(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::rint(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::absolute(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::sigmoid(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::erf(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::flip(
    const Tensor& /* tensor */,
    const unsigned /* dim */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::clip(
    const Tensor& /* tensor */,
    const Tensor& /* low */,
    const Tensor& /* high */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::roll(
    const Tensor& /* tensor */,
    const int /* shift */,
    const unsigned /* axis */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::isnan(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::isinf(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::sign(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::tril(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::triu(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::where(
    const Tensor& /* condition */,
    const Tensor& /* x */,
    const Tensor& /* y */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

void OneDnnBackend::topk(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* k */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::sort(
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

void OneDnnBackend::sort(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::argsort(
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

/************************** Binary Operators ***************************/
#define FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, TYPE)                              \
  Tensor OneDnnBackend::FUNC(const Tensor& /* a */, TYPE /* rhs */) {         \
    throw std::runtime_error(                                                 \
        "OneDnnBackend::" + std::string(#FUNC) + " unimplemented for type " + \
        std::string(#TYPE));                                                  \
  }                                                                           \
  Tensor OneDnnBackend::FUNC(TYPE /* lhs */, const Tensor& /* a */) {         \
    throw std::runtime_error(                                                 \
        "OneDnnBackend::" + std::string(#FUNC) + " unimplemented for type " + \
        std::string(#TYPE));                                                  \
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
#define FL_AF_BINARY_OP_DEF(OP, FUNC)                     \
  Tensor OneDnnBackend::FUNC(                             \
      const Tensor& /* lhs */, const Tensor& /* rhs */) { \
    throw std::runtime_error(                             \
        "OneDnnBackend::" + std::string(#FUNC) +          \
        " unimplemented for two-Tensor inputs.");         \
  }                                                       \
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

Tensor OneDnnBackend::minimum(
    const Tensor& /* lhs */,
    const Tensor& /* rhs */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::maximum(
    const Tensor& /* lhs */,
    const Tensor& /* rhs */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::power(const Tensor& /* lhs */, const Tensor& /* rhs */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

/************************** BLAS ***************************/

Tensor OneDnnBackend::matmul(
    const Tensor& /* lhs */,
    const Tensor& /* rhs */,
    MatrixProperty /* lhsProp */,
    MatrixProperty /* rhsProp */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

/************************** Reductions ***************************/

Tensor OneDnnBackend::amin(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::amax(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

void OneDnnBackend::min(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

void OneDnnBackend::max(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::sum(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::cumsum(
    const Tensor& /* input */,
    const unsigned /* axis */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::argmax(
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::argmin(
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::mean(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::median(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::var(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* bias */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::std(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::norm(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    double /* p */ /* = 2 */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::countNonzero(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::any(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::all(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

void OneDnnBackend::print(const Tensor& tensor) {
  std::cout << "OneDnnTensor" << std::endl
            << tensor.getAdapter<OneDnnTensor>().toString() << std::endl;
}

} // namespace fl
