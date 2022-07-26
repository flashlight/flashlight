/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"

#include <cstring>
#include <numeric>
#include <stdexcept>

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnBackend.h"
#include "flashlight/fl/tensor/backend/onednn/Utils.h"

#include <dnnl_debug.h>

#define FL_ONEDNN_TENSOR_UNIMPLEMENTED \
  throw std::invalid_argument(         \
      "OneDnnTensor::" + std::string(__func__) + " - unimplemented.");

namespace fl {

OneDnnTensor::OneDnnTensor(const Shape& shape, dnnl::memory&& memory)
    : memory_(std::move(memory)), shape_(shape) {}

OneDnnTensor::OneDnnTensor()
    : OneDnnTensor({0}, fl::dtype::f32, nullptr, Location::Host) {}

OneDnnTensor::OneDnnTensor(
    const Shape& shape,
    fl::dtype type,
    const void* ptr,
    Location memoryLocation) : shape_(shape) {
  // TODO handle Location::Device once we add CL support
  if (memoryLocation != Location::Host) {
    throw std::invalid_argument(
        "[OneDnnTensor] initialization data must be on host.");
  }
  const auto memDesc = dnnl::memory::desc(
      detail::shapeToOneDnnDims(shape_),
      detail::flToOneDnnType(type),
      detail::shapeToOneDnnStrides(shape_),
      /* allowEmpty */ true);
  memory_ = dnnl::memory(memDesc, OneDnnBackend::getInstance().engine());
  const auto numDataBytes = shape_.elements() * fl::getTypeSize(type);
  // NOTE, once we support CL, we can take ownership directly for device ptr.
  if (ptr != nullptr) {
    std::memcpy(memory_.get_data_handle(), ptr, numDataBytes);
  }
}

OneDnnTensor::OneDnnTensor(
    const Dim /* nRows */,
    const Dim /* nCols */,
    const Tensor& /* values */,
    const Tensor& /* rowIdx */,
    const Tensor& /* colIdx */,
    StorageType /* storageType */) {
  throw std::runtime_error(
      "OneDnnTensor currently doesn't support sparse tensor");
}

std::unique_ptr<TensorAdapterBase> OneDnnTensor::clone() const {
  // shallow copy the underlying memory
  return std::make_unique<OneDnnTensor>(shape_, dnnl::memory(memory_));
}

Tensor OneDnnTensor::copy() {
  // TODO copy on write
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

Tensor OneDnnTensor::shallowCopy() {
  // shallow copy the underlying memory
  return fl::toTensor<OneDnnTensor>(shape_, dnnl::memory(memory_));
}

TensorBackendType OneDnnTensor::backendType() const {
  return backend().backendType();
}

TensorBackend& OneDnnTensor::backend() const {
  return OneDnnBackend::getInstance();
}

const Shape& OneDnnTensor::shape() {
  return shape_;
}

fl::dtype OneDnnTensor::type() {
  return detail::oneDnnToFlType(memory_.get_desc().data_type());
}

bool OneDnnTensor::isSparse() {
  return false;
}

Location OneDnnTensor::location() {
  return memory_.get_engine().get_kind() == dnnl::engine::kind::cpu
      ? Location::Host
      : Location::Device;
}

void OneDnnTensor::scalar(void* /* out */) {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

void OneDnnTensor::device(void** /* out */) {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

void OneDnnTensor::host(void* /* out */) {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

void OneDnnTensor::unlock() {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

bool OneDnnTensor::isLocked() {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

bool OneDnnTensor::isContiguous() {
  if (shape_.ndim() == 0) { // scalar
    return true;
  }
  const auto& dims = shape_.get();
  const auto leadingStride =
      std::accumulate(dims.begin(), dims.end() - 1, 1, std::multiplies<Dim>());
  return this->strides().get().back() == leadingStride;
}

Shape OneDnnTensor::strides() {
  const auto& memoryDesc = memory_.get_desc().data;
  if (memoryDesc.format_kind != dnnl_format_kind_t::dnnl_blocked) {
    throw std::invalid_argument(
        "[OneDnnTensor::strides] Unexpected memory format kind: " +
        std::string(dnnl_fmt_kind2str(memoryDesc.format_kind)));
  }
  const auto& blockingDesc = memoryDesc.format_desc.blocking;
  std::vector<Dim> strides; // reverse internal strides to get col-major strides
  for (int i = memoryDesc.ndims - 1; i >= 0; i--) {
    strides.push_back(blockingDesc.strides[i]);
  }
  return Shape(strides);
}

const Stream& OneDnnTensor::stream() const {
  return OneDnnBackend::getInstance().stream();
}

Tensor OneDnnTensor::astype(const dtype /* type */) {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

Tensor OneDnnTensor::index(const std::vector<Index>& /* indices */) {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

Tensor OneDnnTensor::flatten() const {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

Tensor OneDnnTensor::flat(const Index& /* idx */) const {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

Tensor OneDnnTensor::asContiguousTensor() {
  // TODO
  // WE won't have strided tensors for now; update this after adding indexing
  if (!isContiguous()) {
    throw std::runtime_error(
        "[OneDnnTensor::asContiguousTensor] Strided tensor currently unsupported");
  }
  return this->copy();
}

void OneDnnTensor::setContext(void* /* context */) {
  // no-op
}

void* OneDnnTensor::getContext() {
  return nullptr;
}

std::string OneDnnTensor::toString() {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

std::ostream& OneDnnTensor::operator<<(std::ostream& ostr) {
  return ostr << toString();
}

/******************** Assignment Operators ********************/
#define FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, TYPE)            \
  void OneDnnTensor::OP(const TYPE& /* val */) {             \
    throw std::invalid_argument(                             \
        "OneDnnTensor::" + std::string(#OP) + " for type " + \
        std::string(#TYPE));                                 \
  }

#define FL_ONEDNN_TENSOR_ASSIGN_OP(OP)                 \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, Tensor);         \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, double);         \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, float);          \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, int);            \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, unsigned);       \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, bool);           \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, char);           \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, unsigned char);  \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, short);          \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, unsigned short); \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, long);           \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, unsigned long);  \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, long long);      \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, unsigned long long);

FL_ONEDNN_TENSOR_ASSIGN_OP(assign); // =
FL_ONEDNN_TENSOR_ASSIGN_OP(inPlaceAdd); // +=
FL_ONEDNN_TENSOR_ASSIGN_OP(inPlaceSubtract); // -=
FL_ONEDNN_TENSOR_ASSIGN_OP(inPlaceMultiply); // *=
FL_ONEDNN_TENSOR_ASSIGN_OP(inPlaceDivide); // /=
#undef FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE
#undef FL_ONEDNN_TENSOR_ASSIGN_OP

} // namespace fl
