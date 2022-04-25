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
#include <cstdlib>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

namespace fl {
namespace {

using reduceFunc_t = af::array (*)(const af::array&, const int);

template <typename T = reduceFunc_t>
af::array afReduceAxes(
    const af::array& input,
    const std::vector<int>& axes,
    T func,
    const bool keepDims = false) {
  auto arr = input;
  for (int dim : axes) {
    arr = func(arr, dim);
  }
  return fl::detail::condenseIndices(arr, keepDims);
}

unsigned
getReducedNumDims(unsigned inSize, unsigned axisSize, const bool keepDims) {
  if (keepDims) {
    return inSize;
  } else {
    if (inSize < axisSize) {
      return 0;
    } else {
      return inSize - axisSize;
    }
  }
}

bool isAllAxisReduction(const Tensor& input, const std::vector<int>& axes) {
  if (input.ndim() == 0 || axes.empty()) {
    return true;
  }
  if (input.ndim() != axes.size()) {
    return false;
  }
  // Check that all dims are present
  auto _axes = axes;
  std::sort(_axes.begin(), _axes.end());
  for (size_t i = 0; i < _axes.size(); ++i) {
    if (_axes[i] != i) {
      return false;
    }
  }
  return true;
}

bool canBroadcast(const Shape& lhs, const Shape& rhs) {
  unsigned nDim = std::max(lhs.ndim(), rhs.ndim());

  for (unsigned i = 0; i < nDim; ++i) {
    if (i + 1 > lhs.ndim() || i + 1 > rhs.ndim()) {
      // One Shape has more dimensions than the other - will broadcast to the
      // smaller tensor
      continue;
    }
    if (lhs[i] != rhs[i] && lhs[i] != 1 && rhs[i] != 1) {
      return false;
    }
  }
  return true;
}

// A binary operation on two ArrayFire arrays
using binaryOpFunc_t =
    af::array (*)(const af::array& lhs, const af::array& rhs);

Tensor doBinaryOpOrBroadcast(
    const Tensor& lhs,
    const Tensor& rhs,
    binaryOpFunc_t func) {
  // Dims are the same or scalar <> 1-el tensor - no broadcasting
  if (lhs.shape() == rhs.shape() ||
      (lhs.elements() <= 1 && rhs.elements() <= 1)) {
    return toTensor<ArrayFireTensor>(
        func(toArray(lhs), toArray(rhs)), lhs.ndim());
  }

  if (canBroadcast(lhs.shape(), rhs.shape())) {
    return toTensor<ArrayFireTensor>(
        af::batchFunc(toArray(lhs), toArray(rhs), func),
        std::max(lhs.ndim(), rhs.ndim()));
  } else {
    std::stringstream ss;
    ss << "doBinaryOpOrBroadcast: cannot perform operation "
          "or broadcasting with tensors of shapes "
       << lhs.shape() << " and " << rhs.shape() << " - dimension mismatch.";
    throw std::invalid_argument(ss.str());
  }
}

} // namespace

ArrayFireBackend::ArrayFireBackend() {
  AF_CHECK(af_init());
}

ArrayFireBackend& ArrayFireBackend::getInstance() {
  static ArrayFireBackend instance;
  return instance;
}

TensorBackendType ArrayFireBackend::backendType() const {
  return TensorBackendType::ArrayFire;
}

/* -------------------------- Compute Functions -------------------------- */

void ArrayFireBackend::sync() {
  af::sync();
}

void ArrayFireBackend::sync(const int deviceId) {
  af::sync(deviceId);
}

void ArrayFireBackend::eval(const Tensor& tensor) {
  af::eval(toArray(tensor));
}

int ArrayFireBackend::getDevice() {
  return af::getDevice();
}

void ArrayFireBackend::setDevice(const int deviceId) {
  af::setDevice(deviceId);
}

int ArrayFireBackend::getDeviceCount() {
  return af::getDeviceCount();
}

bool ArrayFireBackend::supportsDataType(const fl::dtype& dtype) const {
  switch (dtype) {
    case fl::dtype::f16:
      return af::isHalfAvailable(af::getDevice()) &&
          // f16 isn't [yet] supported with the CPU backend per onednn
          // limitations
          !FL_BACKEND_CPU;
    default:
      return true;
  }
}

/* -------------------------- Rand Functions -------------------------- */

void ArrayFireBackend::setSeed(const int seed) {
  af::setSeed(seed);
}

Tensor ArrayFireBackend::randn(const Shape& shape, dtype type) {
  return toTensor<ArrayFireTensor>(
      af::randn(detail::flToAfDims(shape), detail::flToAfType(type)),
      shape.ndim());
}

Tensor ArrayFireBackend::rand(const Shape& shape, dtype type) {
  return toTensor<ArrayFireTensor>(
      af::randu(detail::flToAfDims(shape), detail::flToAfType(type)),
      shape.ndim());
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
#define AF_BACKEND_CREATE_FUN_LITERAL_DEF(TYPE)                          \
  Tensor ArrayFireBackend::fromScalar(TYPE value, const dtype type) {    \
    return toTensor<ArrayFireTensor>(                                    \
        af::constant(value, af::dim4(1), detail::flToAfType(type)),      \
        /* ndim = */ 0);                                                 \
  }                                                                      \
  Tensor ArrayFireBackend::full(                                         \
      const Shape& shape, TYPE value, const dtype type) {                \
    return toTensor<ArrayFireTensor>(                                    \
        af::constant(                                                    \
            value, detail::flToAfDims(shape), detail::flToAfType(type)), \
        shape.ndim());                                                   \
  }
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const double&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const float&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const int&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const char&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned char&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const long&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned long&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const long long&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned long long&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const bool&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const short&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned short&);

Tensor ArrayFireBackend::identity(const Dim dim, const dtype type) {
  return toTensor<ArrayFireTensor>(
      af::identity({dim, dim}, detail::flToAfType(type)), /* numDims = */ 2);
}

Tensor ArrayFireBackend::arange(
    const Shape& shape,
    const Dim seqDim,
    const dtype type) {
  return toTensor<ArrayFireTensor>(
      af::range(detail::flToAfDims(shape), seqDim, detail::flToAfType(type)),
      shape.ndim());
}

Tensor ArrayFireBackend::iota(
    const Shape& dims,
    const Shape& tileDims,
    const dtype type) {
  return toTensor<ArrayFireTensor>(
      af::iota(
          detail::flToAfDims(dims),
          detail::flToAfDims(tileDims),
          detail::flToAfType(type)),
      /* numDims = */ std::max(dims.ndim(), tileDims.ndim()));
}

/************************ Shaping and Indexing *************************/
Tensor ArrayFireBackend::reshape(const Tensor& tensor, const Shape& shape) {
  return toTensor<ArrayFireTensor>(
      af::moddims(toArray(tensor), detail::flToAfDims(shape)), shape.ndim());
}

Tensor ArrayFireBackend::transpose(
    const Tensor& tensor,
    const Shape& axes /* = {} */) {
  if (tensor.ndim() == 1) {
    return tensor;
  } else if (
      tensor.ndim() == 2 && (axes.ndim() == 0 || axes == Shape({1, 0}))) {
    // fastpath for matrices
    return toTensor<ArrayFireTensor>(
        af::transpose(toArray(tensor)), tensor.ndim());
  } else if (axes.ndim() == 0) {
    std::vector<Dim> dims(AF_MAX_DIMS);
    std::iota(std::begin(dims), std::end(dims), 0);
    // Compute the reversed dimensions for as many ndims as are in the input
    for (unsigned i = 0; i < tensor.ndim(); ++i) {
      dims[i] = tensor.ndim() - 1 - i;
    }

    // flip all dimensions
    return toTensor<ArrayFireTensor>(
        af::reorder(toArray(tensor), dims[0], dims[1], dims[2], dims[3]),
        tensor.ndim());
  } else {
    if (axes.ndim() > AF_MAX_DIMS) {
      throw std::invalid_argument(
          "ArrayFire tensor transpose was given "
          "permutation dims with > 4 axes");
    }
    if (axes.ndim() != tensor.ndim()) {
      throw std::invalid_argument(
          "ArrayFire tensor transpose axes don't match tensor's for "
          "permutation - axes must have the same number of "
          "dimensions as the tensor");
    }
    // reorder based on specified dimensions
    std::vector<dim_t> d(AF_MAX_DIMS);
    std::iota(std::begin(d), std::end(d), 0);
    for (size_t i = 0; i < axes.ndim(); ++i) {
      if (axes[i] > tensor.ndim() - 1) {
        throw std::invalid_argument(
            "ArrayFireBackend::transpose - given dimension is larger "
            "than the number of dimensions in the tensor");
      }

      d[i] = axes[i];
    }
    return toTensor<ArrayFireTensor>(
        af::reorder(toArray(tensor), d[0], d[1], d[2], d[3]), tensor.ndim());
  }
}

Tensor ArrayFireBackend::tile(const Tensor& tensor, const Shape& shape) {
  return toTensor<ArrayFireTensor>(
      af::tile(toArray(tensor), detail::flToAfDims(shape)),
      // TODO: check
      std::max(tensor.ndim(), shape.ndim()));
}

Tensor ArrayFireBackend::concatenate(
    const std::vector<Tensor>& tensors,
    const unsigned axis) {
  af::array out;
  switch (tensors.size()) {
    case 0:
      return toTensor<ArrayFireTensor>(ArrayFireTensor()); // empty tensor
    case 1:
      return tensors.front();
    case 2:
      out = af::join(axis, toArray(tensors[0]), toArray(tensors[1]));
      break;
    case 3:
      out = af::join(
          axis, toArray(tensors[0]), toArray(tensors[1]), toArray(tensors[2]));
      break;
    case 4:
      out = af::join(
          axis,
          toArray(tensors[0]),
          toArray(tensors[1]),
          toArray(tensors[2]),
          toArray(tensors[3]));
      break;
    default:
      // TODO: iteratively concat to remove this limitation
      throw std::invalid_argument(
          "ArrayFire concatenate doesn't support > 4 tensors");
  }

  unsigned numDims = tensors[0].ndim();
  if (axis > std::max(numDims - 1, 0u)) {
    numDims = axis + 1;
  }

  // All tensors have the same numdims else AF would throw
  return toTensor<ArrayFireTensor>(std::move(out), numDims);
}

Tensor ArrayFireBackend::nonzero(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(
      af::where(toArray(tensor)), /* numDims = */ 1);
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

  return toTensor<ArrayFireTensor>(
      af::pad(
          toArray(input),
          beginPadding,
          endPadding,
          detail::flToAfPadType(type)),
      /* numDims = */ // TODO: check
      std::max(input.ndim(), static_cast<int>(padWidths.size())));
}

/************************** Unary Operators ***************************/

Tensor ArrayFireBackend::exp(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::exp(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::log(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::log(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::negative(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(-toArray(tensor), tensor.ndim());
}

Tensor ArrayFireBackend::logicalNot(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(!toArray(tensor), tensor.ndim());
}

Tensor ArrayFireBackend::log1p(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::log1p(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sin(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::sin(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::cos(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::cos(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sqrt(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::sqrt(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::tanh(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::tanh(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::floor(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::floor(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::ceil(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::ceil(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::rint(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::round(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::absolute(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::abs(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sigmoid(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::sigmoid(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::erf(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::erf(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::flip(const Tensor& tensor, const unsigned dim) {
  return toTensor<ArrayFireTensor>(
      af::flip(toArray(tensor), dim), tensor.ndim());
}

Tensor ArrayFireBackend::clip(
    const Tensor& tensor,
    const Tensor& low,
    const Tensor& high) {
  return toTensor<ArrayFireTensor>(
      af::clamp(toArray(tensor), toArray(low), toArray(high)), tensor.ndim());
}

Tensor ArrayFireBackend::roll(
    const Tensor& tensor,
    const int shift,
    const unsigned axis) {
  if (axis > AF_MAX_DIMS) {
    throw std::invalid_argument(
        "ArrayFireBackend::roll - given axis > 3 - unsupported");
  }
  std::vector<Dim> shifts(AF_MAX_DIMS, 0);
  shifts[axis] = shift;
  return toTensor<ArrayFireTensor>(
      af::shift(toArray(tensor), shifts[0], shifts[1], shifts[2], shifts[3]),
      tensor.ndim());
}

Tensor ArrayFireBackend::isnan(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::isNaN(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::isinf(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::isInf(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sign(const Tensor& tensor) {
  auto wSigned = 1 - 2 * af::sign(toArray(tensor));
  wSigned(toArray(tensor) == 0) = 0;
  return toTensor<ArrayFireTensor>(std::move(wSigned), tensor.ndim());
}

Tensor ArrayFireBackend::tril(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(
      af::lower(toArray(tensor), /* is_unit_diag = */ false), tensor.ndim());
}

Tensor ArrayFireBackend::triu(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(
      af::upper(toArray(tensor), /* is_unit_diag = */ false), tensor.ndim());
}

Tensor ArrayFireBackend::where(
    const Tensor& condition,
    const Tensor& x,
    const Tensor& y) {
  Tensor orig = x;
  af::replace(toArray(orig), toArray(condition), toArray(y));
  return orig;
}

void ArrayFireBackend::topk(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned k,
    const Dim axis,
    const SortMode sortMode) {
  if (axis != 0) {
    throw std::invalid_argument(
        "ArrayFireTensor topk: operation only supported along zero axis.");
  }
  af::array valuesArr, indicesArr;
  af::topk(
      valuesArr,
      indicesArr,
      toArray(input),
      k,
      axis,
      detail::flToAfTopKSortMode(sortMode));

  values = toTensor<ArrayFireTensor>(std::move(valuesArr), input.ndim());
  indices = toTensor<ArrayFireTensor>(std::move(indicesArr), input.ndim());
}

Tensor ArrayFireBackend::sort(
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode) {
  if (sortMode != SortMode::Descending && sortMode != SortMode::Ascending) {
    throw std::invalid_argument(
        "Cannot sort ArrayFire tensor with given SortMode: "
        "only Descending and Ascending supported.");
  }

  af::array values, indices;
  af::sort(
      values, indices, toArray(input), axis, sortMode == SortMode::Ascending);
  return toTensor<ArrayFireTensor>(std::move(values), input.ndim());
}

void ArrayFireBackend::sort(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode) {
  if (sortMode != SortMode::Descending && sortMode != SortMode::Ascending) {
    throw std::invalid_argument(
        "Cannot sort ArrayFire tensor with given SortMode: "
        "only Descending and Ascending supported.");
  }

  af::array _values, _indices;
  af::sort(
      _values, _indices, toArray(input), axis, sortMode == SortMode::Ascending);
  values = toTensor<ArrayFireTensor>(std::move(_values), input.ndim());
  indices = toTensor<ArrayFireTensor>(std::move(_indices), input.ndim());
}

Tensor ArrayFireBackend::argsort(
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode) {
  if (sortMode != SortMode::Descending && sortMode != SortMode::Ascending) {
    throw std::invalid_argument(
        "Cannot sort ArrayFire tensor with given SortMode: "
        "only Descending and Ascending supported.");
  }

  af::array values, indices;
  af::sort(
      values, indices, toArray(input), axis, sortMode == SortMode::Ascending);
  return toTensor<ArrayFireTensor>(std::move(indices), input.ndim());
}

/************************** Binary Operators ***************************/
// For ArrayFire, af::array already implements overloads for all needed
// operators -- use these by default.
#define FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, TYPE)                   \
  Tensor ArrayFireBackend::FUNC(const Tensor& a, TYPE rhs) {       \
    return toTensor<ArrayFireTensor>(toArray(a) OP rhs, a.ndim()); \
  }                                                                \
  Tensor ArrayFireBackend::FUNC(TYPE lhs, const Tensor& a) {       \
    return toTensor<ArrayFireTensor>(lhs OP toArray(a), a.ndim()); \
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
    return doBinaryOpOrBroadcast(lhs, rhs, af::operator OP);            \
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
  return doBinaryOpOrBroadcast(lhs, rhs, af::min);
}

Tensor ArrayFireBackend::maximum(const Tensor& lhs, const Tensor& rhs) {
  return doBinaryOpOrBroadcast(lhs, rhs, af::max);
}

Tensor ArrayFireBackend::power(const Tensor& lhs, const Tensor& rhs) {
  return doBinaryOpOrBroadcast(lhs, rhs, af::pow);
}

/************************** BLAS ***************************/

Tensor ArrayFireBackend::matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp) {
  unsigned numDims = std::max(lhs.ndim(), rhs.ndim());
  if ((lhs.ndim() == 1 || rhs.ndim() == 1) && numDims > 1) {
    numDims -= 1;
  }

  af::array lhsArray = toArray(lhs);
  af::array rhsArray = toArray(rhs);

  if (lhs.ndim() == 1 && rhs.ndim() == 1) {
    // Simulate a dot product by transpoing the lhs:
    // (1, k) x (k, 1) --> (1, 1) --> reshape to (1)
    // Ignore other transposes since 1D tensors are the transpose of themselves.
    // ArrayFire would otherwise transpose a (k) tensor to (1, k) since (k) =
    // (k, 1, 1, 1) and ArrayFire transpose transposes the first two dimensions.
    lhsProp = MatrixProperty::Transpose;
    rhsProp = MatrixProperty::None;
    numDims = 1;
  } else {
    if (rhs.ndim() == 1) {
      rhsArray = af::moddims(toArray(rhs), {rhs.dim(0), 1});
    }
    if (lhs.ndim() == 1) {
      lhsArray = af::moddims(toArray(lhs), {1, lhs.dim(0)});
    }
  }

  auto arr = af::matmul(
      lhsArray,
      rhsArray,
      detail::flToAfMatrixProperty(lhsProp),
      detail::flToAfMatrixProperty(rhsProp));

  if (lhs.ndim() == 1 && rhs.ndim() == 2) {
    arr = af::moddims(arr, arr.dims(1));
  }

  return toTensor<ArrayFireTensor>(std::move(arr), numDims);
}

/************************** Reductions ***************************/

Tensor ArrayFireBackend::amin(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::min<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::min(af::min(af::min(af::min(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), axes, af::min, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::amax(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::max<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::max(af::max(af::max(af::max(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), axes, af::max, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

void ArrayFireBackend::min(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  af::min(toArray(values), toArray(indices), toArray(input), axis);
  values = toTensor<ArrayFireTensor>(
      detail::condenseIndices(toArray(values), keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
  indices = toTensor<ArrayFireTensor>(
      detail::condenseIndices(toArray(indices), keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
}

void ArrayFireBackend::max(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  af::max(toArray(values), toArray(indices), toArray(input), axis);
  values = toTensor<ArrayFireTensor>(
      detail::condenseIndices(toArray(values), keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
  indices = toTensor<ArrayFireTensor>(
      detail::condenseIndices(toArray(indices), keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
}

Tensor ArrayFireBackend::sum(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::sum<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::sum(af::sum(af::sum(af::sum(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), axes, af::sum, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::cumsum(const Tensor& input, const unsigned axis) {
  return toTensor<ArrayFireTensor>(
      af::accum(toArray(input), axis), /* numDims = */ input.ndim());
}

Tensor ArrayFireBackend::argmax(
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  af::array tmpVal, indices;
  af::max(tmpVal, indices, toArray(input), axis);
  return toTensor<ArrayFireTensor>(
      detail::condenseIndices(indices, keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
}

Tensor ArrayFireBackend::argmin(
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  af::array tmpVal, indices;
  af::min(tmpVal, indices, toArray(input), axis);
  return toTensor<ArrayFireTensor>(
      detail::condenseIndices(indices, keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
}

Tensor ArrayFireBackend::mean(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::mean<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::mean(af::mean(af::mean(af::mean(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes<af::array(const af::array&, const dim_t)>(
            toArray(input), axes, af::mean, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::median(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::median<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    double median = af::median<double>(toArray(input));
    return toTensor<ArrayFireTensor>(
        af::constant(median, 1),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes<af::array(const af::array&, const dim_t)>(
            toArray(input), axes, af::median, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::var(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool bias,
    const bool keepDims) {
  // Use ArrayFire default for one dimension which may be optimized
  auto& arr = toArray(input);
  // Reduce along all axes returning a singleton tensor
  // TODO: modify this to af::var<af::array> to take advantage of the
  // ArrayFire reduce_all kernels once available
  if (isAllAxisReduction(input, axes)) {
    double out = af::var<double>(toArray(input), bias);
    return toTensor<ArrayFireTensor>(af::constant(out, 1), /* numDims = */ 0);
  } else if (axes.size() == 1) {
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(af::var(arr, bias, axes[0]), keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  } else {
    auto meanArr = mean(input, axes, /* keepDims = */ true);
    auto x = af::batchFunc(arr, toArray(meanArr), af::operator-);

    x = af::pow(x, 2);
    x = afReduceAxes(x, axes, af::sum, /* keepDims = */ true);

    int denominator = 1;
    auto dims = arr.dims();
    for (auto dim : axes) {
      denominator *= dims[dim];
    }
    if (bias) {
      denominator--;
    }

    x = x / denominator;
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(x, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::std(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  // TODO: add a bias parameter and `bias ? AF_VARIANCE_SAMPLE :
  // AF_VARIANCE_POPULATION` when requiring to a minimum ArrayFire version that
  // has updated variance and stdev functions
  if (isAllAxisReduction(input, axes)) {
    // TODO: update to af::stdev<af::array> once specialization is available
    double out = af::stdev<double>(toArray(input));
    return toTensor<ArrayFireTensor>(af::constant(out, 1), /* numDims = */ 0);
  } else if (axes.size() == 1) {
    // Use arrayfire default for one dimension which may be optimized
    // TODO: update this? stddev is deprecated.
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(af::stdev(toArray(input), axes[0]), keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
  return this->sqrt(this->var(input, axes, /* bias = */ false, keepDims));
}

Tensor ArrayFireBackend::norm(
    const Tensor& input,
    const std::vector<int>& axes,
    double p /* = 2 */,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // TODO: update to af::norm<af::array> if device-side specialization is
    // available. Either that or use the all-axis specializations with the below
    // implementation
    auto result = af::pow(af::abs(af::flat(toArray(input))), p);
    // Replace with af::sum<af::array>
    result = af::sum(af::sum(af::sum(result)));
    result = af::pow(result, 1 / p);
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(result), /* numDims = */ 0);
  } else {
    auto result = af::pow(af::abs(toArray(input)), p);
    result = afReduceAxes(result, axes, af::sum, keepDims);
    result = af::pow(result, 1 / p);
    return toTensor<ArrayFireTensor>(
        std::move(result),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::countNonzero(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  auto& arr = toArray(input);
  unsigned numDims;
  af::array out;
  if (isAllAxisReduction(input, axes)) {
    out = detail::condenseIndices(
        af::sum(af::sum(af::sum(af::count(arr)))), keepDims);
    numDims = 0;
  } else if (axes.size() == 1) {
    out = af::count(arr, axes.front());
    numDims = getReducedNumDims(input.ndim(), axes.size(), keepDims);
  } else {
    out = afReduceAxes(
        af::count(arr, axes.front()),
        std::vector<int>(axes.begin() + 1, axes.end()),
        af::sum,
        keepDims);
    numDims = getReducedNumDims(input.ndim(), axes.size(), keepDims);
  }
  return toTensor<ArrayFireTensor>(
      detail::condenseIndices(out, keepDims), numDims);
}

Tensor ArrayFireBackend::any(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::anyTrue<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::anyTrue(af::anyTrue(af::anyTrue(af::anyTrue(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), axes, af::anyTrue, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::all(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::allTrue<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::allTrue(af::allTrue(af::allTrue(af::allTrue(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), axes, af::allTrue, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

void ArrayFireBackend::print(const Tensor& tensor) {
  af::print("ArrayFireTensor", toArray(tensor));
}
} // namespace fl
