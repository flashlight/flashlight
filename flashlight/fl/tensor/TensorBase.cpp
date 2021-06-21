/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
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

Tensor::Tensor() : impl_(detail::getDefaultAdapter()) {}

const Shape& Tensor::shape() const {
  return impl_->shape();
}

dtype Tensor::type() const {
  return impl_->type();
}

Tensor Tensor::astype(const dtype type) {
  return impl_->astype(type);
}

Tensor Tensor::operator()(const std::vector<Index>& indices) const {
  return impl_->index(indices);
}

Tensor Tensor::flatten() const {
  return impl_->flatten();
}

TensorBackendType Tensor::backendType() const {
  return impl_->backendType();
}

TensorBackend& Tensor::backend() const {
  return impl_->backend();
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
#undef EXPAND_MACRO_FUNCTION_TYPE
#undef EXPAND_MACRO_FUNCTION

/******************** Assignment Operators ********************/
#define ASSIGN_OP_TYPE(OP, FUN, TYPE)   \
  Tensor& Tensor::OP(const TYPE& val) { \
    impl_->FUN(val);                    \
    return *this;                       \
  }
#define ASSIGN_OP(OP, FUN)                 \
  ASSIGN_OP_TYPE(OP, FUN, Tensor);         \
  ASSIGN_OP_TYPE(OP, FUN, double);         \
  ASSIGN_OP_TYPE(OP, FUN, float);          \
  ASSIGN_OP_TYPE(OP, FUN, int);            \
  ASSIGN_OP_TYPE(OP, FUN, unsigned);       \
  ASSIGN_OP_TYPE(OP, FUN, bool);           \
  ASSIGN_OP_TYPE(OP, FUN, char);           \
  ASSIGN_OP_TYPE(OP, FUN, unsigned char);  \
  ASSIGN_OP_TYPE(OP, FUN, short);          \
  ASSIGN_OP_TYPE(OP, FUN, unsigned short); \
  ASSIGN_OP_TYPE(OP, FUN, long);           \
  ASSIGN_OP_TYPE(OP, FUN, unsigned long);  \
  ASSIGN_OP_TYPE(OP, FUN, long long);      \
  ASSIGN_OP_TYPE(OP, FUN, unsigned long long);

// (operator, function name on impl)
ASSIGN_OP(operator=, assign);
ASSIGN_OP(operator+=, inPlaceAdd);
ASSIGN_OP(operator-=, inPlaceSubtract);
ASSIGN_OP(operator*=, inPlaceMultiply);
ASSIGN_OP(operator/=, inPlaceDivide);
#undef ASSIGN_OP_TYPE
#undef ASSIGN_OP

/* --------------------------- Tensor Operators --------------------------- */

/************************ Shaping and Indexing *************************/

Tensor reshape(const Tensor& tensor, const Shape& shape) {
  return tensor.backend().reshape(tensor, shape);
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

Tensor absolute(const Tensor& tensor) {
  return tensor.backend().absolute(tensor);
}

Tensor clip(const Tensor& tensor, const Tensor& low, const Tensor& high) {
  FL_TENSOR_BACKENDS_MATCH_CHECK(tensor, low, high);
  return tensor.backend().clip(tensor, low, high);
}

Tensor clip(const Tensor& tensor, const Tensor& low, const double& high) {
  return clip(tensor, low, full(tensor.shape(), high));
}

Tensor clip(const Tensor& tensor, const double& low, const Tensor& high) {
  return clip(tensor, full(tensor.shape(), low), high);
}

Tensor clip(const Tensor& tensor, const double& low, const double& high) {
  return clip(tensor, full(tensor.shape(), low), full(tensor.shape(), high));
}

Tensor isnan(const Tensor& tensor) {
  return tensor.backend().isnan(tensor);
}

/************************** Binary Operators ***************************/

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

/************************** Reductions ***************************/

Tensor amin(const Tensor& input, const std::vector<int>& axes) {
  return input.backend().amin(input, axes);
}

Tensor amax(const Tensor& input, const std::vector<int>& axes) {
  return input.backend().amax(input, axes);
}

Tensor sum(const Tensor& input, const std::vector<int>& axes) {
  return input.backend().sum(input, axes);
}

Tensor mean(const Tensor& input, const std::vector<int>& axes) {
  return input.backend().mean(input, axes);
}

Tensor var(const Tensor& input, const std::vector<int>& axes, bool bias) {
  return input.backend().var(input, axes, bias);
}

// fl::var<T>(const Tensor&)
#define GENERATE_VAR(TYPE)                                      \
  template <>                                                   \
  TYPE var(const Tensor& input, bool bias) {                    \
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

double norm(const Tensor& input) {
  return input.backend().norm(input);
}

template <typename T>
T amin(const Tensor& input) {
  return static_cast<T>(input.backend().amin(input));
}

/************************** Utilities ***************************/

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
  if (a.shape().elements() == 0 || b.shape().elements() == 0) {
    return false;
  }
  return fl::amax<double>(fl::abs(a - b)) < absTolerance;
}

} // namespace fl
