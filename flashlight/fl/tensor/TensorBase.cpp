/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorBase.h"

#include <stdexcept>
#include <utility>

#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/TensorBackend.h"

#define FL_TENSOR_BACKENDS_MATCH_CHECK(...)             \
  if (!detail::areBackendsEqual(__VA_ARGS__)) {         \
    throw std::invalid_argument(                        \
        std::string(__func__) +                         \
        " called with tensors of different backends."); \
  }

namespace fl {

Tensor::Tensor(std::unique_ptr<TensorAdapterBase> adapter)
    : impl_(std::move(adapter)) {}

Tensor::~Tensor() {}

Tensor::Tensor(const Tensor& tensor) : impl_(tensor.impl_->clone()) {}

Tensor::Tensor(Tensor&& other) noexcept : impl_(std::move(other.impl_)) {}

Tensor::Tensor() : impl_(detail::getDefaultAdapter()) {}

Tensor::Tensor(
    const Shape& shape,
    fl::dtype type,
    void* ptr,
    MemoryLocation memoryLocation)
    : impl_(detail::getDefaultAdapter(shape, type, ptr, memoryLocation)) {}

Tensor::Tensor(
    const Dim nRows,
    const Dim nCols,
    const Tensor& values,
    const Tensor& rowIdx,
    const Tensor& colIdx,
    StorageType storageType)
    : impl_(detail::getDefaultAdapter(
          nRows,
          nCols,
          values,
          rowIdx,
          colIdx,
          storageType)) {}

Tensor::Tensor(const Shape& shape, fl::dtype type /* = fl::dtype::f32 */)
    : impl_(detail::getDefaultAdapter(shape, type)) {}

Tensor::Tensor(fl::dtype type)
    : impl_(detail::getDefaultAdapter(Shape(), type)) {}

Tensor Tensor::copy() const {
  return impl_->copy();
}

Tensor Tensor::shallowCopy() const {
  return impl_->shallowCopy();
}

const Shape& Tensor::shape() const {
  return impl_->shape();
}

Location Tensor::location() const {
  return impl_->location();
}

size_t Tensor::size() const {
  return impl_->shape().elements();
}

Dim Tensor::dim(const size_t dim) const {
  return shape().dim(dim);
}

size_t Tensor::ndim() const {
  return shape().ndim();
}

bool Tensor::isEmpty() const {
  return size() == 0;
}

size_t Tensor::bytes() const {
  return size() * getTypeSize(type());
}

dtype Tensor::type() const {
  return impl_->type();
}

bool Tensor::isSparse() const {
  return impl_->isSparse();
}

Tensor Tensor::astype(const dtype type) const {
  return impl_->astype(type);
}

Tensor Tensor::operator()(const std::vector<Index>& indices) const {
  return impl_->index(indices);
}

Tensor Tensor::flatten() const {
  return impl_->flatten();
}

Tensor Tensor::flat(const Index& idx) const {
  return impl_->flat(idx);
}

Tensor Tensor::asContiguousTensor() const {
  return impl_->asContiguousTensor();
}

TensorBackendType Tensor::backendType() const {
  return impl_->backendType();
}

TensorBackend& Tensor::backend() const {
  return impl_->backend();
}

#define FL_CREATE_MEMORY_OPS(TYPE)                                          \
  template <>                                                               \
  TYPE Tensor::scalar() const {                                             \
    if (type() != dtype_traits<TYPE>::fl_type) {                            \
      throw std::invalid_argument(                                          \
          "Tensor::scalar: requested type of " +                            \
          std::string(dtype_traits<TYPE>::getName()) +                      \
          " doesn't match tensor type, which is " + dtypeToString(type())); \
    }                                                                       \
    TYPE out;                                                               \
    impl_->scalar(&out);                                                    \
    return out;                                                             \
  }                                                                         \
                                                                            \
  template <>                                                               \
  TYPE* Tensor::device() const {                                            \
    TYPE* out;                                                              \
    void** addr = reinterpret_cast<void**>(&out);                           \
    impl_->device(addr);                                                    \
    return out;                                                             \
  }                                                                         \
                                                                            \
  template <>                                                               \
  TYPE* Tensor::host() const {                                              \
    TYPE* out = reinterpret_cast<TYPE*>(new char[bytes()]);                 \
    impl_->host(out);                                                       \
    return out;                                                             \
  }                                                                         \
                                                                            \
  template <>                                                               \
  void Tensor::host(TYPE* ptr) const {                                      \
    impl_->host(ptr);                                                       \
  }
FL_CREATE_MEMORY_OPS(int);
FL_CREATE_MEMORY_OPS(unsigned);
FL_CREATE_MEMORY_OPS(char);
FL_CREATE_MEMORY_OPS(unsigned char);
FL_CREATE_MEMORY_OPS(long);
FL_CREATE_MEMORY_OPS(unsigned long);
FL_CREATE_MEMORY_OPS(long long);
FL_CREATE_MEMORY_OPS(unsigned long long);
FL_CREATE_MEMORY_OPS(double);
FL_CREATE_MEMORY_OPS(float);
FL_CREATE_MEMORY_OPS(short);
FL_CREATE_MEMORY_OPS(unsigned short);
// void specializations
template <>
void* Tensor::device() const {
  void* out;
  impl_->device(&out);
  return out;
}

template <>
void* Tensor::host() const {
  void* out = reinterpret_cast<void*>(new char[bytes()]);
  impl_->host(out);
  return out;
}

template <>
void Tensor::host(void* ptr) const {
  impl_->host(ptr);
}
#undef FL_CREATE_MEMORY_OPS

void Tensor::unlock() const {
  impl_->unlock();
}

bool Tensor::isLocked() const {
  return impl_->isLocked();
}

bool Tensor::isContiguous() const {
  return impl_->isContiguous();
}

Shape Tensor::strides() const {
  return impl_->strides();
}

void Tensor::setContext(void* context) {
  impl_->setContext(context);
}

void* Tensor::getContext() const {
  return impl_->getContext();
}

std::ostream& Tensor::operator<<(std::ostream& ostr) const {
  return impl_->operator<<(ostr);
}

// Generate template specializations for functions that return types
#define EXPAND_MACRO_FUNCTION_TYPE(FUN, TYPE)             \
  template <>                                             \
  TYPE FUN(const Tensor& input) {                         \
    return static_cast<TYPE>(input.backend().FUN(input)); \
  }
#define EXPAND_MACRO_FUNCTION(FUN)                     \
  EXPAND_MACRO_FUNCTION_TYPE(FUN, int);                \
  EXPAND_MACRO_FUNCTION_TYPE(FUN, unsigned);           \
  EXPAND_MACRO_FUNCTION_TYPE(FUN, char);               \
  EXPAND_MACRO_FUNCTION_TYPE(FUN, unsigned char);      \
  EXPAND_MACRO_FUNCTION_TYPE(FUN, long);               \
  EXPAND_MACRO_FUNCTION_TYPE(FUN, unsigned long);      \
  EXPAND_MACRO_FUNCTION_TYPE(FUN, long long);          \
  EXPAND_MACRO_FUNCTION_TYPE(FUN, unsigned long long); \
  EXPAND_MACRO_FUNCTION_TYPE(FUN, double);             \
  EXPAND_MACRO_FUNCTION_TYPE(FUN, float);              \
  EXPAND_MACRO_FUNCTION_TYPE(FUN, short);              \
  EXPAND_MACRO_FUNCTION_TYPE(FUN, unsigned short);

// fl::amin<T>(const Tensor&)
EXPAND_MACRO_FUNCTION(amin);
// fl::amax<T>(const Tensor&)
EXPAND_MACRO_FUNCTION(amax);
// fl::sum<T>(const Tensor&)
EXPAND_MACRO_FUNCTION(sum);
// fl::mean<T>(const Tensor&)
EXPAND_MACRO_FUNCTION(mean);
// fl::mean<T>(const Tensor&)
EXPAND_MACRO_FUNCTION(median);
#undef EXPAND_MACRO_FUNCTION_TYPE
#undef EXPAND_MACRO_FUNCTION

/******************** Assignment Operators ********************/
#define FL_ASSIGN_OP_TYPE(OP, FUN, TYPE) \
  Tensor& Tensor::OP(TYPE val) {         \
    impl_->FUN(val);                     \
    return *this;                        \
  }
#define FL_ASSIGN_OP(OP, FUN)                        \
  FL_ASSIGN_OP_TYPE(OP, FUN, const Tensor&);         \
  FL_ASSIGN_OP_TYPE(OP, FUN, const double&);         \
  FL_ASSIGN_OP_TYPE(OP, FUN, const float&);          \
  FL_ASSIGN_OP_TYPE(OP, FUN, const int&);            \
  FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned&);       \
  FL_ASSIGN_OP_TYPE(OP, FUN, const bool&);           \
  FL_ASSIGN_OP_TYPE(OP, FUN, const char&);           \
  FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned char&);  \
  FL_ASSIGN_OP_TYPE(OP, FUN, const short&);          \
  FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned short&); \
  FL_ASSIGN_OP_TYPE(OP, FUN, const long&);           \
  FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned long&);  \
  FL_ASSIGN_OP_TYPE(OP, FUN, const long long&);      \
  FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned long long&);

// (operator, function name on impl)
FL_ASSIGN_OP(operator=, assign);
FL_ASSIGN_OP(operator+=, inPlaceAdd);
FL_ASSIGN_OP(operator-=, inPlaceSubtract);
FL_ASSIGN_OP(operator*=, inPlaceMultiply);
FL_ASSIGN_OP(operator/=, inPlaceDivide);
#undef FL_ASSIGN_OP_TYPE
#undef FL_ASSIGN_OP

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
#define FL_FULL_FUN_DEF(TYPE)                                    \
  template <>                                                    \
  Tensor full(const Shape& dims, TYPE value, const dtype type) { \
    return Tensor().backend().full(dims, value, type);           \
  }
FL_FULL_FUN_DEF(const double&);
FL_FULL_FUN_DEF(const float&);
FL_FULL_FUN_DEF(const int&);
FL_FULL_FUN_DEF(const unsigned&);
FL_FULL_FUN_DEF(const char&);
FL_FULL_FUN_DEF(const unsigned char&);
FL_FULL_FUN_DEF(const long&);
FL_FULL_FUN_DEF(const unsigned long&);
FL_FULL_FUN_DEF(const long long&);
FL_FULL_FUN_DEF(const unsigned long long&);
FL_FULL_FUN_DEF(const bool&);
FL_FULL_FUN_DEF(const short&);
FL_FULL_FUN_DEF(const unsigned short&);
#undef FL_FULL_FUN_DEF

Tensor identity(const Dim dim, const dtype type) {
  return Tensor().backend().identity(dim, type);
}

#define FL_ARANGE_FUN_DEF(TYPE)                                             \
  template <>                                                               \
  Tensor arange(TYPE start, TYPE end, TYPE step, const dtype type) {        \
    return fl::arange({static_cast<long>((end - start) / step)}, 0, type) * \
        step +                                                              \
        start;                                                              \
  }
FL_ARANGE_FUN_DEF(const double&);
FL_ARANGE_FUN_DEF(const float&);
FL_ARANGE_FUN_DEF(const int&);
FL_ARANGE_FUN_DEF(const unsigned&);
FL_ARANGE_FUN_DEF(const long&);
FL_ARANGE_FUN_DEF(const unsigned long&);
FL_ARANGE_FUN_DEF(const long long&);
FL_ARANGE_FUN_DEF(const unsigned long long&);

Tensor arange(const Shape& shape, const Dim seqDim, const dtype type) {
  return Tensor().backend().arange(shape, seqDim, type);
}

Tensor iota(const Shape& dims, const Shape& tileDims, const dtype type) {
  return Tensor().backend().iota(dims, tileDims, type);
}

/************************ Shaping and Indexing *************************/

Tensor reshape(const Tensor& tensor, const Shape& shape) {
  return tensor.backend().reshape(tensor, shape);
}

Tensor transpose(const Tensor& tensor, const Shape& axes /* = {} */) {
  return tensor.backend().transpose(tensor, axes);
}

Tensor tile(const Tensor& tensor, const Shape& shape) {
  return tensor.backend().tile(tensor, shape);
}

Tensor concatenate(const std::vector<Tensor>& tensors, unsigned axis) {
  if (tensors.empty()) {
    throw std::invalid_argument("concatenate: called on empty set of tensors");
  }

  // Check all backends match
  TensorBackendType b = tensors.front().backendType();
  const bool matches =
      std::all_of(tensors.begin(), tensors.end(), [b](const Tensor& t) {
        return t.backendType() == b;
      });
  if (!matches) {
    throw std::invalid_argument(
        "concatenate: tried to concatenate tensors of different backends");
  }

  return tensors.front().backend().concatenate(tensors, axis);
}

Tensor nonzero(const Tensor& tensor) {
  return tensor.backend().nonzero(tensor);
}

Tensor pad(
    const Tensor& input,
    const std::vector<std::pair<int, int>>& padWidths,
    const PadType type) {
  return input.backend().pad(input, padWidths, type);
}

/************************** Unary Operators ***************************/
Tensor exp(const Tensor& tensor) {
  return tensor.backend().exp(tensor);
}

Tensor log(const Tensor& tensor) {
  return tensor.backend().log(tensor);
}

Tensor negative(const Tensor& tensor) {
  return tensor.backend().negative(tensor);
}

Tensor logicalNot(const Tensor& tensor) {
  return tensor.backend().logicalNot(tensor);
}

Tensor log1p(const Tensor& tensor) {
  return tensor.backend().log1p(tensor);
}

Tensor sin(const Tensor& tensor) {
  return tensor.backend().sin(tensor);
}

Tensor cos(const Tensor& tensor) {
  return tensor.backend().cos(tensor);
}

Tensor sqrt(const Tensor& tensor) {
  return tensor.backend().sqrt(tensor);
}

Tensor tanh(const Tensor& tensor) {
  return tensor.backend().tanh(tensor);
}

Tensor floor(const Tensor& tensor) {
  return tensor.backend().floor(tensor);
}

Tensor ceil(const Tensor& tensor) {
  return tensor.backend().ceil(tensor);
}

Tensor absolute(const Tensor& tensor) {
  return tensor.backend().absolute(tensor);
}

Tensor sigmoid(const Tensor& tensor) {
  return tensor.backend().sigmoid(tensor);
}

Tensor erf(const Tensor& tensor) {
  return tensor.backend().erf(tensor);
}

Tensor clip(const Tensor& tensor, const Tensor& low, const Tensor& high) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(tensor, low, high);
  return tensor.backend().clip(tensor, low, high);
}

Tensor clip(const Tensor& tensor, const Tensor& low, const double& high) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(tensor, low);
  return clip(tensor, low, full(tensor.shape(), high));
}

Tensor clip(const Tensor& tensor, const double& low, const Tensor& high) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(tensor, high);
  return clip(tensor, full(tensor.shape(), low), high);
}

Tensor clip(const Tensor& tensor, const double& low, const double& high) {
  return clip(tensor, full(tensor.shape(), low), full(tensor.shape(), high));
}

Tensor isnan(const Tensor& tensor) {
  return tensor.backend().isnan(tensor);
}

Tensor isinf(const Tensor& tensor) {
  return tensor.backend().isinf(tensor);
}

Tensor sign(const Tensor& tensor) {
  return tensor.backend().sign(tensor);
}

Tensor tril(const Tensor& tensor) {
  return tensor.backend().tril(tensor);
}

Tensor triu(const Tensor& tensor) {
  return tensor.backend().triu(tensor);
}

Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(condition, x, y);
  return condition.backend().where(condition, x, y);
}

Tensor where(const Tensor& condition, const Tensor& x, const double& y) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(condition, x);
  return condition.backend().where(
      condition, x, fl::full(condition.shape(), y, x.type()));
}

Tensor where(const Tensor& condition, const double& x, const Tensor& y) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(condition, y);
  return condition.backend().where(
      condition, fl::full(condition.shape(), x, y.type()), y);
}

void topk(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned k,
    const Dim axis,
    const SortMode sortMode) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(values, indices, input);
  input.backend().topk(values, indices, input, k, axis, sortMode);
}

Tensor sort(const Tensor& input, const Dim axis, const SortMode sortMode) {
  return input.backend().sort(input, axis, sortMode);
}

Tensor argsort(const Tensor& input, const Dim axis, const SortMode sortMode) {
  return input.backend().argsort(input, axis, sortMode);
}

/************************** Binary Operators ***************************/
#define FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, TYPE) \
  Tensor FUNC(TYPE lhs, const Tensor& rhs) {          \
    return rhs.backend().FUNC(lhs, rhs);              \
  }                                                   \
  Tensor FUNC(const Tensor& lhs, TYPE rhs) {          \
    return lhs.backend().FUNC(lhs, rhs);              \
  }                                                   \
  Tensor operator OP(TYPE lhs, const Tensor& rhs) {   \
    return FUNC(lhs, rhs);                            \
  }                                                   \
  Tensor operator OP(const Tensor& lhs, TYPE rhs) {   \
    return FUNC(lhs, rhs);                            \
  }

#define FL_BINARY_OP_LITERALS_DEF(OP, FUNC)                           \
  FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const bool&);               \
  FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const int&);                \
  FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const unsigned&);           \
  FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const char&);               \
  FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const unsigned char&);      \
  FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const long&);               \
  FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const unsigned long&);      \
  FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const long long&);          \
  FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const unsigned long long&); \
  FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const double&);             \
  FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const float&);              \
  FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const short&);              \
  FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const unsigned short&);

#define FL_BINARY_OP_DEF(OP, FUNC)                           \
  Tensor FUNC(const Tensor& lhs, const Tensor& rhs) {        \
    FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);                \
    return lhs.backend().FUNC(lhs, rhs);                     \
  }                                                          \
  Tensor operator OP(const Tensor& lhs, const Tensor& rhs) { \
    return FUNC(lhs, rhs);                                   \
  }                                                          \
  FL_BINARY_OP_LITERALS_DEF(OP, FUNC);

FL_BINARY_OP_DEF(+, add);
FL_BINARY_OP_DEF(-, sub);
FL_BINARY_OP_DEF(*, mul);
FL_BINARY_OP_DEF(/, div);
FL_BINARY_OP_DEF(==, eq);
FL_BINARY_OP_DEF(!=, neq);
FL_BINARY_OP_DEF(<, lessThan);
FL_BINARY_OP_DEF(<=, lessThanEqual);
FL_BINARY_OP_DEF(>, greaterThan);
FL_BINARY_OP_DEF(>=, greaterThanEqual);
FL_BINARY_OP_DEF(||, logicalOr);
FL_BINARY_OP_DEF(&&, logicalAnd);
FL_BINARY_OP_DEF(%, mod);
FL_BINARY_OP_DEF(|, bitwiseOr);
FL_BINARY_OP_DEF(^, bitwiseXor);
FL_BINARY_OP_DEF(<<, lShift);
FL_BINARY_OP_DEF(>>, rShift);

#undef FL_BINARY_OP_DEF
#undef FL_BINARY_OP_LITERALS_DEF
#undef FL_BINARY_OP_LITERAL_TYPE_DEF

Tensor minimum(const Tensor& lhs, const Tensor& rhs) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
  return lhs.backend().minimum(lhs, rhs);
}

Tensor maximum(const Tensor& lhs, const Tensor& rhs) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
  return lhs.backend().maximum(lhs, rhs);
}

Tensor minimum(const Tensor& lhs, const double& rhs) {
  return minimum(lhs, full(lhs.shape(), rhs));
}

Tensor minimum(const double& lhs, const Tensor& rhs) {
  return minimum(full(rhs.shape(), lhs), rhs);
}

Tensor maximum(const Tensor& lhs, const double& rhs) {
  return maximum(lhs, full(lhs.shape(), rhs));
}

Tensor maximum(const double& lhs, const Tensor& rhs) {
  return maximum(full(rhs.shape(), lhs), rhs);
}

Tensor power(const Tensor& lhs, const Tensor& rhs) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
  return lhs.backend().power(lhs, rhs);
}

Tensor power(const Tensor& lhs, const double& rhs) {
  return lhs.backend().power(lhs, full(lhs.shape(), rhs));
}

/******************************* BLAS ********************************/
Tensor matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
  return lhs.backend().matmul(lhs, rhs, lhsProp, rhsProp);
}

/************************** Reductions ***************************/

Tensor amin(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims /* = false */) {
  return input.backend().amin(input, axes, keepDims);
}

Tensor amax(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims /* = false */) {
  return input.backend().amax(input, axes, keepDims);
}

void min(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    bool keepDims) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(values, indices, input);
  return input.backend().min(values, indices, input, axis, keepDims);
}

void max(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    bool keepDims /* = false */) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(values, indices, input);
  return input.backend().max(values, indices, input, axis, keepDims);
}

Tensor sum(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims /* = false */) {
  return input.backend().sum(input, axes, keepDims);
}

Tensor argmax(const Tensor& input, unsigned axis, bool keepDims /* = false */) {
  return input.backend().argmax(input, axis, keepDims);
}

Tensor argmin(const Tensor& input, unsigned axis, bool keepDims /* = false */) {
  return input.backend().argmin(input, axis, keepDims);
}

Tensor mean(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims /* = false */) {
  return input.backend().mean(input, axes, keepDims);
}

Tensor median(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims /* = false */) {
  return input.backend().median(input, axes, keepDims);
}

Tensor var(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool bias,
    bool keepDims /* = false */) {
  return input.backend().var(input, axes, bias, keepDims);
}

// fl::var<T>(const Tensor&)
#define GENERATE_VAR(TYPE)                                      \
  template <>                                                   \
  TYPE var(const Tensor& input, const bool bias) {              \
    return static_cast<TYPE>(input.backend().var(input, bias)); \
  }

GENERATE_VAR(int);
GENERATE_VAR(unsigned);
GENERATE_VAR(char);
GENERATE_VAR(unsigned char);
GENERATE_VAR(long);
GENERATE_VAR(unsigned long);
GENERATE_VAR(long long);
GENERATE_VAR(unsigned long long);
GENERATE_VAR(double);
GENERATE_VAR(float);
GENERATE_VAR(short);
GENERATE_VAR(unsigned short);
#undef GENERATE_VAR

Tensor std(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims /* = false */) {
  return input.backend().std(input, axes, keepDims);
}

double norm(const Tensor& input) {
  return input.backend().norm(input);
}

Tensor countNonzero(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims /* = false */) {
  return input.backend().countNonzero(input, axes, keepDims);
}

Tensor any(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims /* = false */) {
  return input.backend().any(input, axes, keepDims);
}

bool any(const Tensor& input) {
  return input.backend().any(input);
}

Tensor all(
    const Tensor& input,
    const std::vector<int>& axes,
    bool keepDims /* = false */) {
  return input.backend().all(input, axes, keepDims);
}

bool all(const Tensor& input) {
  return input.backend().all(input);
}

template <typename T>
T amin(const Tensor& input) {
  return static_cast<T>(input.backend().amin(input));
}

template <typename T>
T amax(const Tensor& input) {
  return static_cast<T>(input.backend().amax(input));
}

/************************** Utilities ***************************/

std::ostream& operator<<(std::ostream& ostr, const Tensor& t) {
  t.operator<<(ostr);
  return ostr;
}

void print(const Tensor& tensor) {
  tensor.backend().print(tensor);
}

bool allClose(
    const fl::Tensor& a,
    const fl::Tensor& b,
    const double absTolerance) {
  if (a.type() != b.type()) {
    return false;
  }
  if (a.shape() != b.shape()) {
    return false;
  }
  if (a.size() == 0 && b.size() == 0) {
    return true;
  }
  return fl::amax<double>(fl::abs(a - b)) < absTolerance;
}

} // namespace fl
