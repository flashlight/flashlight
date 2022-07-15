/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"

#define FL_ONEDNN_TENSOR_UNIMPLEMENTED \
  throw std::invalid_argument(         \
      "OneDnnTensor::" + std::string(__func__) + " - unimplemented.");

namespace fl {

OneDnnTensor::OneDnnTensor() {}

OneDnnTensor::OneDnnTensor(
    const Shape& /* shape */,
    fl::dtype /* type */,
    const void* /* ptr */,
    Location /* memoryLocation */) {}

OneDnnTensor::OneDnnTensor(
    const Dim /* nRows */,
    const Dim /* nCols */,
    const Tensor& /* values */,
    const Tensor& /* rowIdx */,
    const Tensor& /* colIdx */,
    StorageType /* storageType */) {}

std::unique_ptr<TensorAdapterBase> OneDnnTensor::clone() const {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

Tensor OneDnnTensor::copy() {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

Tensor OneDnnTensor::shallowCopy() {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

TensorBackendType OneDnnTensor::backendType() const {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

TensorBackend& OneDnnTensor::backend() const {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

const Shape& OneDnnTensor::shape() {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

fl::dtype OneDnnTensor::type() {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

bool OneDnnTensor::isSparse() {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

Location OneDnnTensor::location() {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
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
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

Shape OneDnnTensor::strides() {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

const Stream& OneDnnTensor::stream() const {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
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
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

void OneDnnTensor::setContext(void* /* context */) {
  // Used to store arbitrary data on a Tensor - can be a noop.
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

void* OneDnnTensor::getContext() {
  // Used to store arbitrary data on a Tensor - can be a noop.
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

std::string OneDnnTensor::toString() {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

std::ostream& OneDnnTensor::operator<<(std::ostream& /* ostr */) {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
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
