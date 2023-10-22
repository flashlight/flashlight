/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnBackend.h"
#include "flashlight/fl/tensor/backend/onednn/Utils.h"

#include <dnnl_debug.h>
#include <dnnl_types.h>

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

bool floatsEqual(const void* lhs, const void* rhs, unsigned numFloats) {
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

// Canonicalize all integers or fl::end in the index (except for Tensor) into
// non-negative integers, e.g., range(0, -1) --> range(0, axisDim - 1).
// TODO consider moving this to a general util when we have other backends that
// reuse this logic.
Index canonicalizeIndex(const Index& idx, const Dim axisDim) {
  auto canonicalizeDim = [axisDim](const std::optional<Dim>& optDim) {
    if (!optDim.has_value()) {
      return axisDim;
    }
    const auto dim = optDim.value();
    if (dim < -axisDim || axisDim < dim) { // end is exclusive
      std::ostringstream oss;
      oss << "[canonicalizeIndexByShape] dim out of range: dim = " << dim
          << ", axisDim = " << axisDim;
      throw std::invalid_argument(oss.str());
    }
    return dim < 0 ? axisDim + dim : dim;
  };
  switch (idx.type()) {
    case detail::IndexType::Range: {
      const auto& rangeIdx = idx.get<range>();
      auto start = canonicalizeDim(rangeIdx.start());
      auto stride = rangeIdx.stride();
      auto end = canonicalizeDim(rangeIdx.end());
      if (stride != 1) {
        throw std::invalid_argument(
            "[canonicalizeIndexByShape] Current doesn't support index with stride > 1");
      } else if (start >= end) {
        std::ostringstream oss;
        oss << "[canonicalizeIndexByShape] end must be larger than start: end = "
            << end << ", start: " << start;
        throw std::invalid_argument(oss.str());
      }
      return range(canonicalizeDim(start), canonicalizeDim(end), stride);
    }
    case detail::IndexType::Literal: {
      auto literal = canonicalizeDim(idx.get<Dim>());
      if (literal > axisDim) {
        std::ostringstream oss;
        oss << "[canonicalizeIndexByShape] literal index too large: literal = "
            << literal << ", axisDim = " << axisDim;
        throw std::invalid_argument(oss.str());
      }
      return literal;
    }
    case detail::IndexType::Span: {
      return idx;
    }
    case detail::IndexType::Tensor: {
      throw std::invalid_argument(
          "[canonicalizeIndex] Currently does not support Tensor as index");
    }
  }
  throw std::runtime_error("Unexpected IndexType");
}

} // namespace

OneDnnTensor::SharedData::~SharedData() {
  assert(
      !isDevicePtrLocked &&
      "Must unlock device pointer before OneDnnTensor destruction.");
}

OneDnnTensor::OneDnnTensor(
    std::shared_ptr<SharedData> sharedData,
    const Shape& shape,
    const dnnl::memory::desc& memDesc)
    : sharedData_(std::move(sharedData)), shape_(shape), memDesc_(memDesc) {}

void* OneDnnTensor::getOrEvalDataHandle() {
  if (!sharedData_->isDataReady) {
    stream().sync();
    sharedData_->isDataReady = true;
  }
  return sharedData_->memory.get_data_handle();
}

unsigned OneDnnTensor::getSizeInBytes() const {
  // NOTE ideally we should use `dnnl::memory::desc::get_size()`, but for some
  // reason it returns 0 for submemory with non-zero offset, e.g.,
  // `tensor(1:4)`. See https://github.com/oneapi-src/oneDNN/issues/1429
  auto type = memoryDesc().get_data_type();
  auto typeSize = dnnl::memory::data_type_size(type);
  auto numElems = shape_.elements();
  return numElems * typeSize;
}

OneDnnTensor::OneDnnTensor(const Shape& shape, dnnl::memory&& memory) {
  sharedData_ = std::make_shared<SharedData>();
  shape_ = shape;
  memDesc_ = memory.get_desc();
  sharedData_->memory = std::move(memory);
}

OneDnnTensor::OneDnnTensor()
    : OneDnnTensor({0}, fl::dtype::f32, nullptr, Location::Host) {}

OneDnnTensor::OneDnnTensor(
    const Shape& shape,
    fl::dtype type,
    const void* ptr,
    Location memoryLocation)
    : shape_(shape) {
  // TODO handle Location::Device once we add CL support
  if (memoryLocation != Location::Host) {
    throw std::invalid_argument(
        "[OneDnnTensor] initialization data must be on host.");
  }
  memDesc_ = detail::oneDnnContiguousMemDescFromShape(
      shape, detail::flToOneDnnType(type));
  sharedData_ = std::make_shared<SharedData>();
  sharedData_->memory = dnnl::memory(memDesc_, backend().engine());
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
  // TODO copy on write if this is not a view
  auto& srcMem = sharedData_->memory;
  const auto& srcMemDesc = memoryDesc();
  const auto type = srcMemDesc.get_data_type();
  const auto dstMemDesc =
      detail::oneDnnContiguousMemDescFromShape(shape_, type);
  const auto engine = sharedData_->memory.get_engine();
  auto dstMem = dnnl::memory(dstMemDesc, engine);

  // prepare primitive
  // (using reorder in a passthrough sense to generate a new buffer)
  const auto reorderPrimitiveDesc =
      dnnl::reorder::primitive_desc(engine, srcMemDesc, engine, dstMemDesc);
  const auto reorderPrimitive = dnnl::reorder(reorderPrimitiveDesc);

  // execute primitive
  reorderPrimitive.execute(backend().nativeStream(), srcMem, dstMem);
  return std::make_unique<OneDnnTensor>(shape_, std::move(dstMem));
}

Tensor OneDnnTensor::copy() {
  return Tensor(clone());
}

Tensor OneDnnTensor::shallowCopy() {
  // shallow copy the underlying memory
  return Tensor(std::make_unique<OneDnnTensor>(sharedData_, shape_, memDesc_));
}

TensorBackendType OneDnnTensor::backendType() const {
  return backend().backendType();
}

OneDnnBackend& OneDnnTensor::backend() const {
  return OneDnnBackend::getInstance();
}

const Shape& OneDnnTensor::shape() {
  return shape_;
}

fl::dtype OneDnnTensor::type() {
  return detail::oneDnnToFlType(memoryDesc().get_data_type());
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
  if (shape().elements() == 0) {
    throw std::invalid_argument("Cannot call scalar on empty OneDnnTensor");
  }
  const auto& cpuEngine = backend().cpuEngine();

  // prepare memories
  auto& srcMem = memory();
  const auto& srcMemDesc = memoryDesc();
  const auto type = srcMemDesc.get_data_type();
  // dims are strides are the same for scalar (1s),
  // but reorder requires them to have the same # of dimensions
  dnnl::memory::dims scalarDims(srcMemDesc.get_dims().size(), 1);
  dnnl::memory::dims zeroOffsets(srcMemDesc.get_dims().size(), 0);
  const auto srcScalarMemDesc =
      srcMemDesc.submemory_desc(scalarDims, zeroOffsets);
  const dnnl::memory::desc dstMemDesc(scalarDims, type, scalarDims);
  auto dstMem = dnnl::memory(dstMemDesc, cpuEngine, out);

  // prepare primitive
  const auto reorderPrimitiveDesc = dnnl::reorder::primitive_desc(
      srcMem.get_engine(), srcScalarMemDesc, cpuEngine, dstMemDesc);
  const auto reorderPrimitive = dnnl::reorder(reorderPrimitiveDesc);

  // execute primitive
  auto& stream = backend().nativeStream();
  reorderPrimitive.execute(stream, srcMem, dstMem);
  stream.wait();
}

void OneDnnTensor::device(void** out) {
  *out = sharedData_->memory.get_data_handle();
  sharedData_->isDevicePtrLocked = true;
}

void OneDnnTensor::host(void* out) {
  // TODO once we support arbitrary memory layout, we can simply do a reorder to
  // `out` here, where the target memory desc will be column-major & contiguous.
  if (!isContiguous()) {
    asContiguousTensor().host(out);
  } else {
    // despite the "tranposed" internal representation, the physical data are
    // the same
    const auto& mem = memory();
    void* mappedData = mem.map_data();
    std::memcpy(out, mappedData, getSizeInBytes());
    mem.unmap_data(mappedData);
  }
}

void OneDnnTensor::unlock() {
  sharedData_->isDevicePtrLocked = false;
}

bool OneDnnTensor::isLocked() {
  return sharedData_->isDevicePtrLocked;
}

bool OneDnnTensor::isContiguous() {
  const auto& shape = this->shape();
  if (shape.ndim() == 0) { // scalar
    return true;
  }
  const auto& dims = shape.get();
  const auto leadingStride =
      std::accumulate(dims.begin(), dims.end() - 1, 1, std::multiplies<Dim>());
  return this->strides().get().back() == leadingStride;
}

Shape OneDnnTensor::strides() {
  const auto& memoryDesc = this->memoryDesc();
  if (memoryDesc.get_format_kind() != dnnl::memory::format_kind::blocked) {
    throw std::invalid_argument(
        "[OneDnnTensor::strides] Unexpected memory format kind: " +
        std::string(dnnl_fmt_kind2str(
            static_cast<dnnl_format_kind_t>(memoryDesc.get_format_kind()))));
  }
  const auto& _strides = memoryDesc.get_strides();
  std::vector<Dim> strides; // reverse internal strides to get col-major strides
  for (int i = memoryDesc.get_ndims() - 1; i >= 0; i--) {
    strides.push_back(_strides[i]);
  }
  return Shape(strides);
}

const Stream& OneDnnTensor::stream() const {
  return backend().stream();
}

Tensor OneDnnTensor::astype(const dtype type) {
  // prepare memories
  auto& srcMem = sharedData_->memory;
  const auto engine = srcMem.get_engine();
  const auto& srcMemDesc = memoryDesc();
  const auto dstMemDesc = detail::oneDnnContiguousMemDescFromShape(
      shape(), detail::flToOneDnnType(type));
  auto dstMem = dnnl::memory(dstMemDesc, engine);

  // prepare primitive
  const auto reorderPrimitiveDesc =
      dnnl::reorder::primitive_desc(engine, srcMemDesc, engine, dstMemDesc);
  const auto reorderPrimitive = dnnl::reorder(reorderPrimitiveDesc);

  // execute primitive
  reorderPrimitive.execute(backend().nativeStream(), srcMem, dstMem);
  return toTensor<OneDnnTensor>(shape(), std::move(dstMem));
}

Tensor OneDnnTensor::index(const std::vector<Index>& indices) {
  const auto& shape = this->shape();
  // allow indexing scalar with empty indices
  if (indices.size() > shape.ndim() ||
      (indices.size() == 0 && shape.ndim() > 0)) {
    std::ostringstream oss;
    oss << "[OneDnnTensor::index] Invalid number of indices: " << indices.size()
        << " for tensor of ndim = " << shape.ndim();
    throw std::invalid_argument(oss.str());
  }
  // by default, assume all indices are spans
  // recall that shape and dims are in reversed order
  dnnl::memory::dims dims(shape.get().rbegin(), shape.get().rend());
  dnnl::memory::dims offsets(shape.ndim(), 0);
  std::vector<int> dimsAxesWithLiteralIndex;
  for (int shapeAxis = 0; shapeAxis < indices.size(); shapeAxis++) {
    int dimsAxis = shape.ndim() - 1 - shapeAxis;
    const auto idx = canonicalizeIndex(indices[shapeAxis], shape[shapeAxis]);
    switch (idx.type()) {
      case detail::IndexType::Range: {
        const auto& rangeIdx = idx.get<range>();
        offsets[dimsAxis] = rangeIdx.start();
        dims[dimsAxis] = rangeIdx.endVal() - rangeIdx.start();
        continue;
      }
      case detail::IndexType::Span: {
        continue;
      }
      case detail::IndexType::Literal: {
        offsets[dimsAxis] = idx.get<Dim>();
        dims[dimsAxis] = 1;
        dimsAxesWithLiteralIndex.push_back(dimsAxis);
        continue;
      }
      case detail::IndexType::Tensor: {
        throw std::invalid_argument(
            "[OneDnnTensor::index] Currently does not support Tensor as index");
      }
    }
    throw std::runtime_error("Unexpected IndexType");
  }
  const auto condensedDims =
      detail::removeIndices(dims, dimsAxesWithLiteralIndex);
  // recall that scalar has fl::Shape {}, but OneDNN requires at least 1 dim
  const auto subMemDesc = memoryDesc().submemory_desc(dims, offsets);
  const auto resultIsScalar = condensedDims.empty();
  // N.B. these reshapes don't require row-major layout
  const auto indexedMemDesc = resultIsScalar
      ? subMemDesc.reshape({1})
      : subMemDesc.reshape(condensedDims);
  const auto indexedShape = detail::oneDnnDimsToShape(condensedDims);
  return toTensor<OneDnnTensor>(sharedData_, indexedShape, indexedMemDesc);
}

Tensor OneDnnTensor::flatten() const {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

Tensor OneDnnTensor::flat(const Index& /* idx */) const {
  FL_ONEDNN_TENSOR_UNIMPLEMENTED;
}

Tensor OneDnnTensor::asContiguousTensor() {
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
const T*
printData1D(std::ostringstream& oss, const T* elements, const fl::Dim rows) {
  oss << '[';
  for (auto row = 0; row < rows; row++) {
    if (row != 0) { // not first/topmost row
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
  printDataMultiDims(
      oss, static_cast<const T*>(data), shape.get(), shape.ndim());
  oss << std::endl; // make it easier to read
  return oss.str();
}

std::string OneDnnTensor::toString() {
  // TODO lift this up into a util method: Tensor -> std::string
  std::vector<char> vec(getSizeInBytes());
  void* data = vec.data();
  this->host(data);
  const auto& shape = this->shape();
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

#define FL_ONEDNN_TENSOR_ASSIGN_OP_LITERALS(OP)        \
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

#define FL_ONEDNN_TENSOR_ASSIGN_OP(OP)         \
  FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE(OP, Tensor); \
  FL_ONEDNN_TENSOR_ASSIGN_OP_LITERALS(OP)

FL_ONEDNN_TENSOR_ASSIGN_OP_LITERALS(assign); // =
FL_ONEDNN_TENSOR_ASSIGN_OP(inPlaceAdd); // +=
FL_ONEDNN_TENSOR_ASSIGN_OP(inPlaceSubtract); // -=
FL_ONEDNN_TENSOR_ASSIGN_OP(inPlaceMultiply); // *=
FL_ONEDNN_TENSOR_ASSIGN_OP(inPlaceDivide); // /=
#undef FL_ONEDNN_TENSOR_ASSIGN_OP_TYPE
#undef FL_ONEDNN_TENSOR_ASSIGN_OP

void OneDnnTensor::assign(const Tensor& tensor) {
  auto& other = toOneDnnTensor(tensor);
  if (this->sharedData_ == other.sharedData_) {
    return;
  }

  if (this->shape() != other.shape()) {
    throw std::runtime_error("Cannot update OneDNN tensor to different shape");
  }

  // prepare primitive
  auto thisMem = this->memory();
  auto otherMem = other.memory();
  const auto reorderPrimitiveDesc = dnnl::reorder::primitive_desc(
      otherMem.get_engine(),
      other.memoryDesc(),
      thisMem.get_engine(),
      this->memoryDesc());
  const auto reorderPrimitive = dnnl::reorder(reorderPrimitiveDesc);

  // execute primitive
  reorderPrimitive.execute(backend().nativeStream(), otherMem, thisMem);
  this->sharedData_->isDataReady = false;
}

bool OneDnnTensor::equals(OneDnnTensor&& other) {
  // TODO lift this up into a util method: (Tensor, Tensor) -> std::string
  if (this->sharedData_ == other.sharedData_) {
    return true;
  }
  if (this->shape() != other.shape()) {
    return false;
  }
  const auto& thisMemDesc = this->memoryDesc();
  const auto type = thisMemDesc.get_data_type();
  if (type != other.memoryDesc().get_data_type()) {
    return false;
  }
  // TODO investigate ways to speed up this on non-CPU platform.
  std::vector<char> lhsVec(this->getSizeInBytes());
  std::vector<char> rhsVec(other.getSizeInBytes());
  void* lhsData = lhsVec.data();
  void* rhsData = rhsVec.data();
  this->host(lhsData);
  other.host(rhsData);
  // TODO update once f64 is available (after bumping OneDNN to newer version)
  return type == dnnl::memory::data_type::f32
      ? floatsEqual(lhsData, rhsData, this->shape().elements())
      : bytesEqual(lhsData, rhsData, getSizeInBytes());
}

dnnl::memory& OneDnnTensor::memory() {
  return sharedData_->memory;
}

const dnnl::memory::desc& OneDnnTensor::memoryDesc() const {
  return memDesc_;
}

OneDnnTensor& toOneDnnTensor(const Tensor& tensor) {
  auto type = tensor.backendType();
  if (type != TensorBackendType::OneDnn) {
    std::ostringstream oss;
    oss << "[toOneDnnTensor] expected oneDNN-backed tensor, got " << type;
    throw std::invalid_argument(oss.str());
  }
  return tensor.getAdapter<OneDnnTensor>();
}

OneDnnTensor& toOneDnnTensor(Tensor& tensor) {
  return toOneDnnTensor(static_cast<const Tensor&>(tensor));
}

} // namespace fl
