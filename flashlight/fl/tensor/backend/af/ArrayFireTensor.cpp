/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

#include <af/arith.h>
#include <af/device.h>
#include <af/internal.h>

namespace fl {

const af::array& toArray(const Tensor& tensor) {
  if (tensor.backendType() != TensorBackendType::ArrayFire) {
    throw std::invalid_argument("toArray: tensor is not ArrayFire-backed");
  }
  return tensor.getAdapter<ArrayFireTensor>().getHandle();
}

af::array& toArray(Tensor& tensor) {
  if (tensor.backendType() != TensorBackendType::ArrayFire) {
    throw std::invalid_argument("toArray: tensor is not ArrayFire-backed");
  }
  return tensor.getAdapter<ArrayFireTensor>().getHandle();
}

ArrayFireTensor::ArrayFireTensor(af::array&& array)
    : arrayHandle_(std::make_shared<af::array>(std::move(array))) {}

ArrayFireTensor::ArrayFireTensor(
    std::shared_ptr<af::array> arr,
    std::vector<af::index>&& indices)
    : arrayHandle_(arr),
      indices_(std::move(indices)),
      handle_(IndexedArrayComponent()) {}

ArrayFireTensor::ArrayFireTensor(std::shared_ptr<af::array> arr)
    : arrayHandle_(arr) {}

ArrayFireTensor::ArrayFireTensor() : handle_(ArrayComponent()) {}

ArrayFireTensor::ArrayFireTensor(
    const Shape& shape,
    fl::dtype type,
    void* ptr,
    Location memoryLocation)
    : arrayHandle_(std::make_shared<af::array>(
          detail::fromFlData(shape, ptr, type, memoryLocation))),
      handle_(ArrayComponent()) {}

af::array::array_proxy ArrayFireTensor::IndexedArrayComponent::get(
    const ArrayFireTensor& inst) {
  auto& i = inst.indices_.value();
  auto& a = *(inst.arrayHandle_);
  return a(i[0], i[1], i[2], i[3]);
}

af::array& ArrayFireTensor::ArrayComponent::get(const ArrayFireTensor& inst) {
  return *(inst.arrayHandle_);
}

const af::array& ArrayFireTensor::getHandle() const {
  if (!std::holds_alternative<ArrayComponent>(handle_)) {
    throw std::logic_error(
        "ArrayFireTensor::getHandle() - underlying tensor is an array_proxy");
  }
  return *arrayHandle_;
}

af::array& ArrayFireTensor::getHandle() {
  // If the handle currently requires indexing, perform the indexing, change the
  // getter to visit, and clear the indices. Upcast the af::array::array_proxy
  // to an af::array via its operator array() and update the handle.
  // Additionally, since we can't directly mutate the dimensions of an
  // af::array::array_proxy, condense the indices of the resulting array after
  // the conversion.
  if (!std::holds_alternative<ArrayComponent>(handle_)) {
    arrayHandle_ = std::make_shared<af::array>(detail::condenseIndices(
        std::get<IndexedArrayComponent>(handle_).get(*this)));
    handle_ = ArrayComponent(); // set to passthrough
    indices_ = {}; // remove indices
  }
  return *arrayHandle_;
}

std::unique_ptr<TensorAdapterBase> ArrayFireTensor::clone() const {
  af::array arr = getHandle(); // increment internal AF refcount
  return std::unique_ptr<ArrayFireTensor>(new ArrayFireTensor(std::move(arr)));
}

Tensor ArrayFireTensor::copy() {
  return toTensor<ArrayFireTensor>(arrayHandle_->copy());
}

Tensor ArrayFireTensor::shallowCopy() {
  // ensure indexing is resolved so copying a handle ref is sufficient
  getHandle();

  return Tensor(
      std::unique_ptr<ArrayFireTensor>(new ArrayFireTensor(arrayHandle_)));
}

TensorBackendType ArrayFireTensor::backendType() const {
  return TensorBackendType::ArrayFire;
}

TensorBackend& ArrayFireTensor::backend() const {
  // The ArrayFire backend has a single ArrayFireBackend instance per process.
  return ::fl::ArrayFireBackend::getInstance();
}

const Shape& ArrayFireTensor::shape() {
  // Update the Shape in-place. Doesn't change any underlying data; only the
  // mirrored Shape metadata.
  detail::afToFlDims(getHandle().dims(), shape_);
  return shape_;
}

fl::dtype ArrayFireTensor::type() {
  return detail::afToFlType(getHandle().type());
}

void ArrayFireTensor::scalar(void* out) {
  AF_CHECK(af_get_scalar(out, getHandle().get()));
}

void ArrayFireTensor::device(void** out) {
  AF_CHECK(af_get_device_ptr(out, getHandle().get()));
}

void ArrayFireTensor::host(void** out) {
  AF_CHECK(af_get_data_ptr(*out, getHandle().get()));
}

void ArrayFireTensor::unlock() {
  AF_CHECK(af_unlock_array(getHandle().get()));
}

bool ArrayFireTensor::isContiguous() {
  return af::isLinear(getHandle());
}

Tensor ArrayFireTensor::astype(const dtype type) {
  auto a = getHandle().as(detail::flToAfType(type));
  return toTensor<ArrayFireTensor>(std::move(a));
}

Tensor ArrayFireTensor::index(const std::vector<Index>& indices) {
  if (indices.size() > AF_MAX_DIMS) {
    throw std::invalid_argument(
        "ArrayFire-backed tensor was indexed with > 4 elements:"
        "ArrayFire tensors support up to 4 dimensions.");
  }
  std::vector<af::index> afIndices(4, af::span);
  for (size_t i = 0; i < indices.size(); ++i) {
    afIndices[i] = detail::flToAfIndex(indices[i]);
  }
  getHandle(); // if this tensor was a view, run indexing and promote
  return fl::Tensor(std::unique_ptr<ArrayFireTensor>(
      new ArrayFireTensor(arrayHandle_, std::move(afIndices))));
}

Tensor ArrayFireTensor::flatten() const {
  return toTensor<ArrayFireTensor>(af::flat(getHandle()));
}

/******************** Assignment Operators ********************/
#define ASSIGN_OP_TYPE(FUN, AF_OP, TYPE)                                 \
  void ArrayFireTensor::FUN(const TYPE& val) {                           \
    std::visit(                                                          \
        [val, this](auto&& arr) { arr.get(*this) AF_OP val; }, handle_); \
  }
#define ASSIGN_OP(FUN, AF_OP)                                                  \
  void ArrayFireTensor::FUN(const Tensor& tensor) {                            \
    std::visit(                                                                \
        [&tensor, this](auto&& arr) { arr.get(*this) AF_OP toArray(tensor); }, \
        handle_);                                                              \
  }                                                                            \
  ASSIGN_OP_TYPE(FUN, AF_OP, double);                                          \
  ASSIGN_OP_TYPE(FUN, AF_OP, float);                                           \
  ASSIGN_OP_TYPE(FUN, AF_OP, int);                                             \
  ASSIGN_OP_TYPE(FUN, AF_OP, unsigned);                                        \
  ASSIGN_OP_TYPE(FUN, AF_OP, bool);                                            \
  ASSIGN_OP_TYPE(FUN, AF_OP, char);                                            \
  ASSIGN_OP_TYPE(FUN, AF_OP, unsigned char);                                   \
  ASSIGN_OP_TYPE(FUN, AF_OP, short);                                           \
  ASSIGN_OP_TYPE(FUN, AF_OP, unsigned short);                                  \
  ASSIGN_OP_TYPE(FUN, AF_OP, long);                                            \
  ASSIGN_OP_TYPE(FUN, AF_OP, unsigned long);                                   \
  ASSIGN_OP_TYPE(FUN, AF_OP, long long);                                       \
  ASSIGN_OP_TYPE(FUN, AF_OP, unsigned long long);

// (function name, AF op). Use build-in AF operators.
ASSIGN_OP(assign, =);
ASSIGN_OP(inPlaceAdd, +=);
ASSIGN_OP(inPlaceSubtract, -=);
ASSIGN_OP(inPlaceMultiply, *=);
ASSIGN_OP(inPlaceDivide, /=);
#undef ASSIGN_OP_TYPE
#undef ASSIGN_OP

/******************** Tensor Creation Functions ********************/
#define AF_FULL_FUN_DEF(TYPE)                                        \
  template <>                                                        \
  Tensor full(const Shape& dims, TYPE value, const dtype type) {     \
    return toTensor<ArrayFireTensor>(af::constant(                   \
        value, detail::flToAfDims(dims), detail::flToAfType(type))); \
  }
AF_FULL_FUN_DEF(const double&);
AF_FULL_FUN_DEF(const float&);
AF_FULL_FUN_DEF(const int&);
AF_FULL_FUN_DEF(const unsigned&);
AF_FULL_FUN_DEF(const char&);
AF_FULL_FUN_DEF(const unsigned char&);
AF_FULL_FUN_DEF(const long&);
AF_FULL_FUN_DEF(const unsigned long&);
AF_FULL_FUN_DEF(const long long&);
AF_FULL_FUN_DEF(const unsigned long long&);
AF_FULL_FUN_DEF(const bool&);
AF_FULL_FUN_DEF(const short&);
AF_FULL_FUN_DEF(const unsigned short&);

Tensor identity(const Dim dim, const dtype type) {
  return toTensor<ArrayFireTensor>(
      af::identity({dim, dim}, detail::flToAfType(type)));
}

/************************** Binary Operators ***************************/
// For ArrayFire, af::array already implements overloads for all needed
// operators -- use these by default.
#define AF_BINARY_OP_TYPE_DEF(OP, TYPE)                  \
  Tensor operator OP(const Tensor& a, TYPE rhs) {        \
    return toTensor<ArrayFireTensor>(toArray(a) OP rhs); \
  }                                                      \
  Tensor operator OP(TYPE lhs, const Tensor& a) {        \
    return toTensor<ArrayFireTensor>(lhs OP toArray(a)); \
  }
#define AF_BINARY_OP_LITERALS_DEF(OP)                   \
  AF_BINARY_OP_TYPE_DEF(OP, const bool&);               \
  AF_BINARY_OP_TYPE_DEF(OP, const int&);                \
  AF_BINARY_OP_TYPE_DEF(OP, const unsigned&);           \
  AF_BINARY_OP_TYPE_DEF(OP, const char&);               \
  AF_BINARY_OP_TYPE_DEF(OP, const unsigned char&);      \
  AF_BINARY_OP_TYPE_DEF(OP, const long&);               \
  AF_BINARY_OP_TYPE_DEF(OP, const unsigned long&);      \
  AF_BINARY_OP_TYPE_DEF(OP, const long long&);          \
  AF_BINARY_OP_TYPE_DEF(OP, const unsigned long long&); \
  AF_BINARY_OP_TYPE_DEF(OP, const double&);             \
  AF_BINARY_OP_TYPE_DEF(OP, const float&);              \
  AF_BINARY_OP_TYPE_DEF(OP, const short&);              \
  AF_BINARY_OP_TYPE_DEF(OP, const unsigned short&);
// Operations on fl::Tensor call the respective operator overloads that are
// already defined on af::arrays
#define AF_BINARY_OP_DEF(OP, FUNC)                                  \
  Tensor FUNC(const Tensor& lhs, const Tensor& rhs) {               \
    return toTensor<ArrayFireTensor>(toArray(lhs) OP toArray(rhs)); \
  }                                                                 \
  AF_BINARY_OP_LITERALS_DEF(OP);
// Definitions
// Since ArrayFire implements operator overloads, map both fl::Tensor
// functions and fl::Tensor operator overloads back to the af::array
// overloads.
AF_BINARY_OP_DEF(+, add);
AF_BINARY_OP_DEF(-, sub);
AF_BINARY_OP_DEF(*, mul);
AF_BINARY_OP_DEF(/, div);
AF_BINARY_OP_DEF(==, eq);
AF_BINARY_OP_DEF(!=, neq);
AF_BINARY_OP_DEF(<, lessThan);
AF_BINARY_OP_DEF(<=, lessThanEqual);
AF_BINARY_OP_DEF(>, greaterThan);
AF_BINARY_OP_DEF(>=, greaterThanEqual);
AF_BINARY_OP_DEF(||, logicalOr);
AF_BINARY_OP_DEF(&&, logicalAnd);
AF_BINARY_OP_DEF(%, mod);
AF_BINARY_OP_DEF(|, bitwiseOr);
AF_BINARY_OP_DEF(^, bitwiseXor);
AF_BINARY_OP_DEF(<<, lShift);
AF_BINARY_OP_DEF(>>, rShift);

} // namespace fl
