/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/stub/StubTensor.h"

#define FL_STUB_TENSOR_UNIMPLEMENTED \
  throw std::invalid_argument(       \
      "StubTensor::" + std::string(__func__) + " - unimplemented.");

namespace fl {

StubTensor::StubTensor() = default;

StubTensor::StubTensor(
    const Shape& /* shape */,
    fl::dtype /* type */,
    const void* /* ptr */,
    Location /* memoryLocation */) {}

StubTensor::StubTensor(
    const Dim /* nRows */,
    const Dim /* nCols */,
    const Tensor& /* values */,
    const Tensor& /* rowIdx */,
    const Tensor& /* colIdx */,
    StorageType /* storageType */) {}

std::unique_ptr<TensorAdapterBase> StubTensor::clone() const {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

Tensor StubTensor::copy() {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

Tensor StubTensor::shallowCopy() {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

TensorBackendType StubTensor::backendType() const {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

TensorBackend& StubTensor::backend() const {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

const Shape& StubTensor::shape() {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

fl::dtype StubTensor::type() {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

bool StubTensor::isSparse() {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

Location StubTensor::location() {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

void StubTensor::scalar(void* /* out */) {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

void StubTensor::device(void** /* out */) {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

void StubTensor::host(void* /* out */) {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

void StubTensor::unlock() {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

bool StubTensor::isLocked() {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

bool StubTensor::isContiguous() {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

Shape StubTensor::strides() {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

const Stream& StubTensor::stream() const {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

Tensor StubTensor::astype(const dtype /* type */) {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

Tensor StubTensor::index(const std::vector<Index>& /* indices */) {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

Tensor StubTensor::flatten() const {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

Tensor StubTensor::flat(const Index& /* idx */) const {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

Tensor StubTensor::asContiguousTensor() {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

void StubTensor::setContext(void* /* context */) {
  // Used to store arbitrary data on a Tensor - can be a noop.
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

void* StubTensor::getContext() {
  // Used to store arbitrary data on a Tensor - can be a noop.
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

std::string StubTensor::toString() {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

std::ostream& StubTensor::operator<<(std::ostream& /* ostr */) {
  FL_STUB_TENSOR_UNIMPLEMENTED;
}

/******************** Assignment Operators ********************/
#define FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, TYPE)            \
  void StubTensor::OP(const TYPE& /* val */) {             \
    throw std::invalid_argument(                           \
        "StubTensor::" + std::string(#OP) + " for type " + \
        std::string(#TYPE));                               \
  }

#define FL_STUB_TENSOR_ASSIGN_OP(OP)                 \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, Tensor);         \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, double);         \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, float);          \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, int);            \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, unsigned);       \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, bool);           \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, char);           \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, unsigned char);  \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, short);          \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, unsigned short); \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, long);           \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, unsigned long);  \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, long long);      \
  FL_STUB_TENSOR_ASSIGN_OP_TYPE(OP, unsigned long long);

FL_STUB_TENSOR_ASSIGN_OP(assign); // =
FL_STUB_TENSOR_ASSIGN_OP(inPlaceAdd); // +=
FL_STUB_TENSOR_ASSIGN_OP(inPlaceSubtract); // -=
FL_STUB_TENSOR_ASSIGN_OP(inPlaceMultiply); // *=
FL_STUB_TENSOR_ASSIGN_OP(inPlaceDivide); // /=
#undef FL_STUB_TENSOR_ASSIGN_OP_TYPE
#undef FL_STUB_TENSOR_ASSIGN_OP

} // namespace fl
