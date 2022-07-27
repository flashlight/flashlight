/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"

#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <sstream>

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnBackend.h"
#include "flashlight/fl/tensor/backend/onednn/Utils.h"

#include <dnnl_debug.h>

#define FL_ONEDNN_TENSOR_UNIMPLEMENTED \
  throw std::invalid_argument(         \
      "OneDnnTensor::" + std::string(__func__) + " - unimplemented.");

namespace fl {

namespace {

constexpr float kfloatEqualTolerance = 1e-5;

template <typename T>
void copyScalar(void* out, const void* data) {
  *(static_cast<T*>(out)) = *(static_cast<const T*>(data));
}

bool floatsEqual(
    const void* lhs,
    const void* rhs,
    unsigned numFloats) {
  auto lhsFloats = static_cast<const float*>(lhs);
  auto rhsFloats = static_cast<const float*>(rhs);
  // TODO consider adding loop parallelism
  for (auto i = 0; i < numFloats; i++) {
    if (std::abs(lhsFloats[i] - rhsFloats[i]) >= kfloatEqualTolerance) {
      return false;
    }
  }
  return true;
}

bool bytesEqual(const void* lhs, const void* rhs, unsigned numBytes) {
  return std::memcmp(lhs, rhs, numBytes) == 0;
}

dnnl::memory::desc copyMemDescWithNewType(
    const dnnl::memory::desc& memDesc,
    const dnnl::memory::data_type newType) {
  dnnl::memory::desc memDescCopy = memDesc;
  memDescCopy.data.data_type = dnnl::memory::convert_to_c(newType);
  return memDescCopy;
}

} // namespace

OneDnnTensor::OneDnnTensor(std::shared_ptr<SharedData> sharedData)
    : sharedData_(std::move(sharedData)) {}

const void* OneDnnTensor::getContiguousData() {
  return isContiguous()
      ? getOrEvalDataHandle()
      : asContiguousTensor().getAdapter<OneDnnTensor>().getOrEvalDataHandle();
}

void* OneDnnTensor::getOrEvalDataHandle() {
  if (!sharedData_->isDataReady) {
    stream().sync();
    sharedData_->isDataReady = true;
  }
  return sharedData_->memory.get_data_handle();
}

OneDnnTensor::OneDnnTensor(const Shape& shape, dnnl::memory&& memory) {
  sharedData_ = std::make_shared<SharedData>();
  sharedData_->shape = shape;
  sharedData_->memory = std::move(memory);
}

OneDnnTensor::OneDnnTensor()
    : OneDnnTensor({0}, fl::dtype::f32, nullptr, Location::Host) {}

OneDnnTensor::OneDnnTensor(
    const Shape& shape,
    fl::dtype type,
    const void* ptr,
    Location memoryLocation) {
  // TODO handle Location::Device once we add CL support
  if (memoryLocation != Location::Host) {
    throw std::invalid_argument(
        "[OneDnnTensor] initialization data must be on host.");
  }
  const auto memDesc = dnnl::memory::desc(
      detail::shapeToOneDnnDims(shape),
      detail::flToOneDnnType(type),
      detail::shapeToOneDnnStrides(shape),
      /* allowEmpty */ true);
  sharedData_ = std::make_shared<SharedData>();
  sharedData_->shape = shape;
  sharedData_->memory = dnnl::memory(memDesc, OneDnnBackend::getInstance().engine());
  const auto numDataBytes = shape.elements() * fl::getTypeSize(type);
  // NOTE, once we support CL, we can take ownership directly for device ptr.
  if (ptr != nullptr) {
    std::memcpy(sharedData_->memory.get_data_handle(), ptr, numDataBytes);
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
  return std::make_unique<OneDnnTensor>(sharedData_);
}

Tensor OneDnnTensor::copy() {
  // TODO copy on write
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

Tensor OneDnnTensor::shallowCopy() {
  // shallow copy the underlying memory
  return Tensor(std::make_unique<OneDnnTensor>(sharedData_));
}

TensorBackendType OneDnnTensor::backendType() const {
  return backend().backendType();
}

TensorBackend& OneDnnTensor::backend() const {
  return OneDnnBackend::getInstance();
}

const Shape& OneDnnTensor::shape() {
  return sharedData_->shape;
}

fl::dtype OneDnnTensor::type() {
  return detail::oneDnnToFlType(sharedData_->memory.get_desc().data_type());
}

bool OneDnnTensor::isSparse() {
  return false;
}

Location OneDnnTensor::location() {
  return sharedData_->memory.get_engine().get_kind() == dnnl::engine::kind::cpu
      ? Location::Host
      : Location::Device;
}

void OneDnnTensor::scalar(void* out) {
  if (sharedData_->shape.elements() == 0) {
    throw std::invalid_argument("Cannot call scalar on empty OneDnnTensor");
  }
  const void* data = getOrEvalDataHandle();
  const auto type = sharedData_->memory.get_desc().data_type();
  switch(type) {
    case dnnl::memory::data_type::f32: return copyScalar<float>(out, data);
    case dnnl::memory::data_type::s32: return copyScalar<int>(out, data);
    case dnnl::memory::data_type::s8: return copyScalar<char>(out, data);
    case dnnl::memory::data_type::u8: return copyScalar<unsigned char>(out, data);
    case dnnl::memory::data_type::undef:
    case dnnl::memory::data_type::f16:
    case dnnl::memory::data_type::bf16:
      throw std::runtime_error(
          "Currently do not support scalarization of type: " +
          detail::oneDnnDataTypeToStr(type));
  }
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
  const auto& shape = sharedData_->shape;
  if (shape.ndim() == 0) { // scalar
    return true;
  }
  const auto& dims = shape.get();
  const auto leadingStride =
      std::accumulate(dims.begin(), dims.end() - 1, 1, std::multiplies<Dim>());
  return this->strides().get().back() == leadingStride;
}

Shape OneDnnTensor::strides() {
  const auto& memoryDesc = sharedData_->memory.get_desc().data;
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

Tensor OneDnnTensor::astype(const dtype type) {
  // prepare memories
  auto& srcMem = sharedData_->memory;
  const auto engine = srcMem.get_engine();
  const auto& srcMemDesc = srcMem.get_desc();
  const auto dstMemDesc =
      copyMemDescWithNewType(srcMemDesc, detail::flToOneDnnType(type));
  auto dstMem = dnnl::memory(dstMemDesc, engine);

  // prepare primitive
  const auto reorderPrimitiveDesc = dnnl::reorder::primitive_desc(
      engine, srcMemDesc, engine, dstMemDesc);
  const auto reorderPrimitive = dnnl::reorder(reorderPrimitiveDesc);

  // execute primitive
  reorderPrimitive.execute(
      OneDnnBackend::getInstance().nativeStream(), srcMem, dstMem);
  return toTensor<OneDnnTensor>(sharedData_->shape, std::move(dstMem));
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

template <typename T>
void printData(std::ostringstream& oss, const T& element) {
  oss << element;
}

// specialization to print out element as number instead of ascii character
template <>
void printData(std::ostringstream& oss, const unsigned char& element) {
  printData<unsigned>(oss, element); // safe cast w/o precision loss
}

// specialization to print out element as number instead of ascii character
template <>
void printData(std::ostringstream& oss, const char& element) {
  printData<int>(oss, element); // safe cast w/o precision loss
}

// Treat `elements` as a column vector and print it to oss in the following
// format:
// ```
// [$(elements[0]),
//  ...,
//  $(elements[rows - 1])]
// ```
// NOTE no newline at the end
// RETURN pointer to the element after the last element to be printed.
template <typename T>
const T* printData1D(
    std::ostringstream& oss,
    const T* elements,
    const fl::Dim rows) {
  oss << '[';
  for (auto row = 0; row < rows; row++) {
    if (row != 0){ // not first/topmost row
      oss << ' ';
    }
    printData(oss, elements[row]);
    if (row != rows - 1) { // not last/bottommost row
      oss << ',' << std::endl;
    }
  }
  oss << ']';
  return elements + rows;
}

// Treat `elements` as a column-major 2D matrix and print it to oss in the
// following format:
// ```
// [[$(elements[0][0]), ..., $(elements[cols-1][0])]
//   ...,
//  [$(elements[0][rows-1]), ..., $(elements[cols-1][rows-1])]]
// ```
// NOTE no newline at the end
// RETURN pointer to the element after the last element to be printed.
template <typename T>
const T* printData2D(
    std::ostringstream& oss,
    const T* elements,
    const fl::Dim rows,
    const fl::Dim cols,
    const unsigned prefixSpaces = 0) {
  oss << '[';
  for (auto row = 0; row < rows; row++) {
    if (row != 0) { // not first/topmost row
      oss << std::string(prefixSpaces + 1, ' ');
    }
    oss << '[';
    for (auto col = 0; col < cols; col++) {
      printData(oss, elements[col * rows + row]);
      if (col != cols - 1) { // not last/rightmost column
        oss << ", ";
      }
    }
    oss << ']';
    if (row != rows - 1) { // not last/bottommost row
      oss << ',' << std::endl;
    }
  }
  oss << ']';
  return elements + (cols * rows);
}

// Treat `elements` as a column-major tensor with dimensions dims[0:dimEndIdx],
// and print it to oss as "slices" of tensors starting from the last dimension.
// e.g., let N = dims[dimEndIdx-1], then we print:
// ```
// [slice0
//   ...,
//  sliceN]
// ```
// with spaces in front of each line, if `dimEndIdx != dims.size()`.
// NOTE no newline at the end
// RETURN pointer to the element after the last element to be printed.
template <typename T>
const T* printDataMultiDims(
    std::ostringstream& oss,
    const T* elements,
    const std::vector<fl::Dim>& dims,
    const unsigned dimEndIdx) { // exclusive index
  if (dimEndIdx == 0) { // scalar
    return printData1D(oss, elements, 1);
  } else if (dimEndIdx == 1) {
    return printData1D(oss, elements, dims[0]);
  } else if (dimEndIdx == 2) {
    const auto prefixSpaces = dims.size() - dimEndIdx;
    return printData2D(oss, elements, dims[0], dims[1], prefixSpaces);
  }
  const auto dimTensors = dims[dimEndIdx - 1];
  const T* nextStart = elements;
  oss << '[';
  for (auto i = 0; i < dimTensors; i++) {
    if (i != 0) {
      const auto prefixSpaces = dims.size() - dimEndIdx + 1;
      oss << std::string(prefixSpaces, ' ');
    }
    nextStart = printDataMultiDims(oss, nextStart, dims, dimEndIdx - 1);
    if (i != dimTensors - 1) { // not last tensor
      oss << ',' << std::endl;
    }
  }
  oss << ']';
  return nextStart;
}

template <typename T>
std::string dataToString(const void* data, const Shape& shape) {
  std::ostringstream oss;
  printDataMultiDims(oss, static_cast<const T*>(data), shape.get(), shape.ndim());
  oss << std::endl; // make it easier to read
  return oss.str();
}

std::string OneDnnTensor::toString() {
  const void* data = getContiguousData();
  const auto& shape = sharedData_->shape;
  switch (type()) {
    case fl::dtype::f16:
      throw std::runtime_error("OneDnnTensor::toString doesn't support f16");
    case fl::dtype::f32:
      return dataToString<float>(data, shape);
    case fl::dtype::f64:
      return dataToString<double>(data, shape);
    case fl::dtype::b8:
      return dataToString<char>(data, shape);
    case fl::dtype::s16:
      return dataToString<short>(data, shape);
    case fl::dtype::s32:
      return dataToString<int>(data, shape);
    case fl::dtype::s64:
      return dataToString<long long>(data, shape);
    case fl::dtype::u8:
      return dataToString<unsigned char>(data, shape);
    case fl::dtype::u16:
      return dataToString<unsigned short>(data, shape);
    case fl::dtype::u32:
      return dataToString<unsigned int>(data, shape);
    case fl::dtype::u64:
      return dataToString<unsigned long long>(data, shape);
  }
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

bool OneDnnTensor::equals(OneDnnTensor&& other) {
  if (this->sharedData_ == other.sharedData_) {
    return true;
  }
  if (this->sharedData_->shape != other.sharedData_->shape) {
    return false;
  }
  const auto& thisMemDesc = this->sharedData_->memory.get_desc();
  const auto type = thisMemDesc.data_type();
  if (type != other.sharedData_->memory.get_desc().data_type()) {
    return false;
  }
  // TODO investigate ways to speed up this on non-CPU platform.
  const void* lhsData = this->getContiguousData();
  const void* rhsData = other.getContiguousData();
  // TODO update once f64 is available (after bumping OneDNN to newer version)
  return type == dnnl::memory::data_type::f32
      ? floatsEqual(lhsData, rhsData, this->sharedData_->shape.elements())
      : bytesEqual(lhsData, rhsData, thisMemDesc.get_size());
}

dnnl::memory& OneDnnTensor::memory() {
  return sharedData_->memory;
}

} // namespace fl
