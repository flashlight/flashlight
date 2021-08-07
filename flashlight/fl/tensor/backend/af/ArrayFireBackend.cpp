/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <af/algorithm.h>
#include <af/arith.h>
#include <af/blas.h>
#include <af/data.h>
#include <af/device.h>
#include <af/gfor.h>
#include <af/lapack.h>
#include <af/random.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

namespace fl {
namespace {

typedef af::array (*reduceFunc_t)(const af::array&, const int);

template <typename T = reduceFunc_t>
af::array afReduceAxes(
    const af::array& input,
    const std::vector<int>& axes,
    T func,
    bool keepDims = false) {
  auto arr = input;
  for (int dim : axes) {
    arr = func(arr, dim);
  }
  return fl::detail::condenseIndices(arr, keepDims);
}

} // namespace

ArrayFireBackend::ArrayFireBackend() {
  AF_CHECK(af_init());
}

ArrayFireBackend& ArrayFireBackend::getInstance() {
  static ArrayFireBackend instance;
  return instance;
}

/* -------------------------- Compute Functions -------------------------- */

void ArrayFireBackend::sync() {
  af::sync();
}

void ArrayFireBackend::sync(int deviceId) {
  af::sync(deviceId);
}

void ArrayFireBackend::eval(const Tensor& tensor) {
  af::eval(toArray(tensor));
}

int ArrayFireBackend::getDevice() {
  return af::getDevice();
}

void ArrayFireBackend::setDevice(int deviceId) {
  af::setDevice(deviceId);
}

/* -------------------------- Rand Functions -------------------------- */

void ArrayFireBackend::setSeed(int seed) {
  af::setSeed(seed);
}

Tensor ArrayFireBackend::randn(const Shape& shape, dtype type) {
  return toTensor<ArrayFireTensor>(
      af::randn(detail::flToAfDims(shape), detail::flToAfType(type)));
}

Tensor ArrayFireBackend::rand(const Shape& shape, dtype type) {
  return toTensor<ArrayFireTensor>(
      af::randu(detail::flToAfDims(shape), detail::flToAfType(type)));
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
#define AF_BACKEND_FULL_FUN_DEF(TYPE)                                \
  Tensor ArrayFireBackend::full(                                     \
      const Shape& dims, TYPE value, const dtype type) {             \
    return toTensor<ArrayFireTensor>(af::constant(                   \
        value, detail::flToAfDims(dims), detail::flToAfType(type))); \
  }
AF_BACKEND_FULL_FUN_DEF(const double&);
AF_BACKEND_FULL_FUN_DEF(const float&);
AF_BACKEND_FULL_FUN_DEF(const int&);
AF_BACKEND_FULL_FUN_DEF(const unsigned&);
AF_BACKEND_FULL_FUN_DEF(const char&);
AF_BACKEND_FULL_FUN_DEF(const unsigned char&);
AF_BACKEND_FULL_FUN_DEF(const long&);
AF_BACKEND_FULL_FUN_DEF(const unsigned long&);
AF_BACKEND_FULL_FUN_DEF(const long long&);
AF_BACKEND_FULL_FUN_DEF(const unsigned long long&);
AF_BACKEND_FULL_FUN_DEF(const bool&);
AF_BACKEND_FULL_FUN_DEF(const short&);
AF_BACKEND_FULL_FUN_DEF(const unsigned short&);

Tensor ArrayFireBackend::identity(const Dim dim, const dtype type) {
  return toTensor<ArrayFireTensor>(
      af::identity({dim, dim}, detail::flToAfType(type)));
}

Tensor ArrayFireBackend::arange(
    const Shape& shape,
    const Dim seqDim,
    const dtype type) {
  return toTensor<ArrayFireTensor>(
      af::range(detail::flToAfDims(shape), seqDim, detail::flToAfType(type)));
}

Tensor ArrayFireBackend::iota(
    const Shape& dims,
    const Shape& tileDims,
    const dtype type) {
  return toTensor<ArrayFireTensor>(af::iota(
      detail::flToAfDims(dims),
      detail::flToAfDims(tileDims),
      detail::flToAfType(type)));
}

/************************ Shaping and Indexing *************************/
Tensor ArrayFireBackend::reshape(const Tensor& tensor, const Shape& shape) {
  return toTensor<ArrayFireTensor>(
      af::moddims(toArray(tensor), detail::flToAfDims(shape)));
}

Tensor ArrayFireBackend::transpose(
    const Tensor& tensor,
    const Shape& dims /* = {} */) {
  if (tensor.shape().ndim() == 2 &&
      (dims.ndim() == 0 || dims == Shape({1, 0}))) {
    // fastpath for matrices
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(af::transpose(toArray(tensor))));
  } else if (dims.ndim() == 0) {
    // flip all dimensions
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(af::reorder(toArray(tensor), 3, 2, 1, 0)));
  } else {
    if (dims.ndim() > AF_MAX_DIMS) {
      throw std::invalid_argument(
          "ArrayFire tensor transpose was given "
          "permutation dims with > 4 axes");
    }
    // reorder based on specified dimensions
    std::vector<dim_t> d(AF_MAX_DIMS);
    std::iota(std::begin(d), std::end(d), 0);
    for (size_t i = 0; i < dims.ndim(); ++i) {
      d[i] = dims[i];
    }
    return toTensor<ArrayFireTensor>(detail::condenseIndices(
        af::reorder(toArray(tensor), d[0], d[1], d[2], d[3])));
  }
}

Tensor ArrayFireBackend::tile(const Tensor& tensor, const Shape& shape) {
  return toTensor<ArrayFireTensor>(
      af::tile(toArray(tensor), detail::flToAfDims(shape)));
}

Tensor ArrayFireBackend::concatenate(
    const std::vector<Tensor>& tensors,
    unsigned axis) {
  if (tensors.size() > 10) {
    throw std::invalid_argument(
        "ArrayFire concatenate doesn't support > 10 tensors");
  }
  std::vector<af_array> arrs(tensors.size());
  std::transform(
      tensors.begin(), tensors.end(), arrs.begin(), [](const Tensor& t) {
        return toArray(t).get();
      });
  af_array handle = nullptr;
  AF_CHECK(af_join_many(&handle, axis, tensors.size(), arrs.data()));
  return toTensor<ArrayFireTensor>(af::array(handle));
}

Tensor ArrayFireBackend::nonzero(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::where(toArray(tensor)));
}

Tensor ArrayFireBackend::pad(
    const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type) {
  if (padWidths.size() > AF_MAX_DIMS) {
    throw std::invalid_argument(
        "ArrayFireBackend::pad - given padWidths for more than 4 dimensions");
  }

  // convert ((begin_1, end_1), ..., (begin_k, end_k)) to ((begin_1, ...,
  // begin_k), (end_1, ..., end_k)) for ArrayFire
  af::dim4 beginPadding, endPadding;
  for (size_t i = 0; i < padWidths.size(); ++i) {
    auto& [first, second] = padWidths[i];
    beginPadding[i] = first;
    endPadding[i] = second;
  }

  return toTensor<ArrayFireTensor>(af::pad(
      toArray(input), beginPadding, endPadding, detail::flToAfPadType(type)));
}

/************************** Unary Operators ***************************/

Tensor ArrayFireBackend::exp(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::exp(toArray(tensor)));
}

Tensor ArrayFireBackend::log(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::log(toArray(tensor)));
}

Tensor ArrayFireBackend::negative(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(-toArray(tensor));
}

Tensor ArrayFireBackend::logicalNot(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(!toArray(tensor));
}

Tensor ArrayFireBackend::log1p(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::log1p(toArray(tensor)));
}

Tensor ArrayFireBackend::sin(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::sin(toArray(tensor)));
}

Tensor ArrayFireBackend::cos(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::cos(toArray(tensor)));
}

Tensor ArrayFireBackend::sqrt(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::sqrt(toArray(tensor)));
}

Tensor ArrayFireBackend::tanh(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::tanh(toArray(tensor)));
}

Tensor ArrayFireBackend::floor(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::floor(toArray(tensor)));
}

Tensor ArrayFireBackend::ceil(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::ceil(toArray(tensor)));
}

Tensor ArrayFireBackend::absolute(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::abs(toArray(tensor)));
}

Tensor ArrayFireBackend::sigmoid(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::sigmoid(toArray(tensor)));
}

Tensor ArrayFireBackend::erf(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::erf(toArray(tensor)));
}

Tensor ArrayFireBackend::clip(
    const Tensor& tensor,
    const Tensor& low,
    const Tensor& high) {
  return toTensor<ArrayFireTensor>(
      af::clamp(toArray(tensor), toArray(low), toArray(high)));
}

Tensor ArrayFireBackend::isnan(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::isNaN(toArray(tensor)));
}

Tensor ArrayFireBackend::isinf(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::isInf(toArray(tensor)));
}

Tensor ArrayFireBackend::sign(const Tensor& tensor) {
  auto wSigned = 1 - 2 * af::sign(toArray(tensor));
  wSigned(toArray(tensor) == 0) = 0;
  return toTensor<ArrayFireTensor>(std::move(wSigned));
}

Tensor ArrayFireBackend::where(
    const Tensor& condition,
    const Tensor& x,
    const Tensor& y) {
  Tensor orig = x;
  af::replace(toArray(orig), toArray(condition), toArray(y));
  return orig;
}

/************************** Binary Operators ***************************/
// For ArrayFire, af::array already implements overloads for all needed
// operators -- use these by default.
#define FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, TYPE)             \
  Tensor ArrayFireBackend::FUNC(const Tensor& a, TYPE rhs) { \
    return toTensor<ArrayFireTensor>(toArray(a) OP rhs);     \
  }                                                          \
  Tensor ArrayFireBackend::FUNC(TYPE lhs, const Tensor& a) { \
    return toTensor<ArrayFireTensor>(lhs OP toArray(a));     \
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
#define FL_AF_BINARY_OP_DEF(OP, FUNC)                                   \
  Tensor ArrayFireBackend::FUNC(const Tensor& lhs, const Tensor& rhs) { \
    return toTensor<ArrayFireTensor>(toArray(lhs) OP toArray(rhs));     \
  }                                                                     \
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
FL_AF_BINARY_OP_DEF(|, bitwiseOr);
FL_AF_BINARY_OP_DEF(^, bitwiseXor);
FL_AF_BINARY_OP_DEF(<<, lShift);
FL_AF_BINARY_OP_DEF(>>, rShift);
#undef FL_AF_BINARY_OP_DEF
#undef FL_AF_BINARY_OP_TYPE_DEF
#undef FL_AF_BINARY_OP_LITERALS_DEF

Tensor ArrayFireBackend::minimum(const Tensor& lhs, const Tensor& rhs) {
  return toTensor<ArrayFireTensor>(af::min(toArray(lhs), toArray(rhs)));
}

Tensor ArrayFireBackend::maximum(const Tensor& lhs, const Tensor& rhs) {
  return toTensor<ArrayFireTensor>(af::max(toArray(lhs), toArray(rhs)));
}

Tensor ArrayFireBackend::power(const Tensor& lhs, const Tensor& rhs) {
  return toTensor<ArrayFireTensor>(af::pow(toArray(lhs), toArray(rhs)));
}

/************************** Matrix Multiply ***************************/

Tensor ArrayFireBackend::matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp) {
  return toTensor<ArrayFireTensor>(af::matmul(
      toArray(lhs),
      toArray(rhs),
      detail::flToAfMatrixProperty(lhsProp),
      detail::flToAfMatrixProperty(rhsProp)));
}

/************************** Reductions ***************************/

Tensor ArrayFireBackend::amin(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims) {
  return toTensor<ArrayFireTensor>(
      afReduceAxes(toArray(input), axes, af::min, keepDims));
}

// TODO: consolidate with above
double ArrayFireBackend::amin(const Tensor& input) {
  return af::min<double>(toArray(input));
}

Tensor ArrayFireBackend::amax(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims) {
  return toTensor<ArrayFireTensor>(
      afReduceAxes(toArray(input), axes, af::max, keepDims));
}

// TODO: consolidate with above
double ArrayFireBackend::amax(const Tensor& input) {
  return af::max<double>(toArray(input));
}

Tensor ArrayFireBackend::sum(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims) {
  return toTensor<ArrayFireTensor>(
      afReduceAxes(toArray(input), axes, af::sum, keepDims));
}

// TODO: consolidate with above
double ArrayFireBackend::sum(const Tensor& input) {
  return af::sum<double>(toArray(input));
}

Tensor ArrayFireBackend::mean(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims) {
  return toTensor<ArrayFireTensor>(
      afReduceAxes<af::array(const af::array&, const dim_t)>(
          toArray(input), axes, af::mean, keepDims));
}

// TODO: consolidate with above
double ArrayFireBackend::mean(const Tensor& input) {
  return af::mean<double>(toArray(input));
}

Tensor ArrayFireBackend::median(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims) {
  return toTensor<ArrayFireTensor>(
      afReduceAxes<af::array(const af::array&, const dim_t)>(
          toArray(input), axes, af::median, keepDims));
}

// TODO: consolidate with above
double ArrayFireBackend::median(const Tensor& input) {
  return af::median<double>(toArray(input));
}

Tensor ArrayFireBackend::var(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool bias,
    bool keepDims) {
  // Use arrayfire default for one dimension which may be optimized
  auto& arr = toArray(input);
  if (axes.size() == 1) {
    return toTensor<ArrayFireTensor>(detail::condenseIndices(
        af::var(
            arr, bias ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION, axes[0]),
        keepDims));
  }
  auto meanArr = mean(input, axes, /* keepDims = */ false);
  // TODO Replace when we have batchFunc for fl::Tensor
  auto x = af::batchFunc(arr, toArray(meanArr), af::operator-);
  x = af::pow(x, 2);
  x = afReduceAxes(x, axes, af::sum);

  int denominator = 1;
  auto dims = toArray(input).dims();
  for (auto dim : axes) {
    denominator *= dims[dim];
  }
  if (bias) {
    denominator--;
  }

  x = x / denominator;
  return toTensor<ArrayFireTensor>(detail::condenseIndices(x, keepDims));
}

// TODO: consolidate with above
double ArrayFireBackend::var(const Tensor& input, const bool bias) {
  return af::var<double>(toArray(input), bias);
}

Tensor ArrayFireBackend::std(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims) {
  if (axes.size() == 1) {
    // Use arrayfire default for one dimension which may be optimized
    return toTensor<ArrayFireTensor>(af::stdev(toArray(input), axes[0]));
  }
  return this->sqrt(this->var(input, axes, /* bias = */ false, keepDims));
}

double ArrayFireBackend::norm(const Tensor& input) {
  return af::norm(toArray(input));
}

Tensor ArrayFireBackend::countNonzero(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims) {
  auto& arr = toArray(input);
  af::array out;
  if (axes.size() == 0) {
    out = af::sum(af::count(arr));
  } else if (axes.size() == 1) {
    out = af::count(arr, axes.front());
  } else {
    out = afReduceAxes(
        af::count(arr, axes.front()),
        std::vector<int>(axes.begin() + 1, axes.end()),
        af::sum,
        keepDims);
  }
  return toTensor<ArrayFireTensor>(detail::condenseIndices(out, keepDims));
}

Tensor ArrayFireBackend::any(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims) {
  return toTensor<ArrayFireTensor>(
      afReduceAxes(toArray(input), axes, af::anyTrue, keepDims));
}

bool ArrayFireBackend::any(const Tensor& input) {
  return af::anyTrue<bool>(toArray(input));
}

Tensor ArrayFireBackend::all(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims) {
  return toTensor<ArrayFireTensor>(
      afReduceAxes(toArray(input), axes, af::allTrue, keepDims));
}

bool ArrayFireBackend::all(const Tensor& input) {
  return af::allTrue<bool>(toArray(input));
}

void ArrayFireBackend::print(const Tensor& tensor) {
  af::print("ArrayFireTensor", toArray(tensor));
}
} // namespace fl
