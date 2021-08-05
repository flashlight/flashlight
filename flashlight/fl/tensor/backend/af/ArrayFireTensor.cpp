/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
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
#include <af/backend.h>
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
  switch (i.size()) {
    case 1:
      return a(i[0]);
    case 2:
      return a(i[0], i[1]);
    case 3:
      return a(i[0], i[1], i[2]);
    case 4:
      return a(i[0], i[1], i[2], i[3]);
    default:
      throw std::invalid_argument(
          "ArrayFireTensor::IndexedArrayComponent::get - "
          "given invalid number of index components.");
  }
}

af::array& ArrayFireTensor::ArrayComponent::get(const ArrayFireTensor& inst) {
  return *(inst.arrayHandle_);
}

const af::array& ArrayFireTensor::getHandle() const {
  return const_cast<ArrayFireTensor*>(this)->getHandle();
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

Location ArrayFireTensor::location() {
  switch (af::getBackendId(getHandle())) {
    case AF_BACKEND_CUDA:
    case AF_BACKEND_OPENCL:
      return Location::Device;
    case AF_BACKEND_CPU:
      return Location::Host;
    default:
      throw std::logic_error(
          "ArrayFireTensor::location got an unmatched location");
  }
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

Shape ArrayFireTensor::strides() {
  // TODO(jacobkahn) do we need to condenseDims here?
  return detail::afToFlDims(af::getStrides(getHandle()));
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

  // If indexing with a single element and it's an Array, don't use spans
  // TODO: vet and stress test this a lot more
  std::vector<af::index> afIndices;
  if (indices.size() == 1 &&
      indices.front().type() == detail::IndexType::Tensor) {
    afIndices = {af::index(0)};
  } else {
    afIndices = {af::span, af::span, af::span, af::span};
  }
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

void ArrayFireTensor::setContext(void* context) {} // noop

void* ArrayFireTensor::getContext() {
  return nullptr; // noop
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

} // namespace fl
