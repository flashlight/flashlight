/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/onednn/OneDnnBackend.h"

#include <cassert>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"
#include "flashlight/fl/tensor/backend/onednn/Utils.h"

#define FL_ONEDNN_BACKEND_UNIMPLEMENTED \
  throw std::invalid_argument(          \
      "OneDnnBackend::" + std::string(__func__) + " - unimplemented.");

namespace fl {

namespace {

dnnl::engine::kind getEngineKind(const Tensor& tensor) {
  return toOneDnnTensor(tensor).memory().get_engine().get_kind();
}

bool hasCpuEngine(const Tensor& tensor) {
  return getEngineKind(tensor) == dnnl::engine::kind::cpu;
}

bool haveSameEngines(const std::vector<const Tensor*>& tensorPtrs) {
  if (tensorPtrs.empty()) {
    return true;
  }
  const auto engineKind = getEngineKind(*tensorPtrs.front());
  for (unsigned i = 1; i < tensorPtrs.size(); i++) {
    if (getEngineKind(*tensorPtrs[i]) != engineKind) {
      return false;
    }
  }
  return true;
}

template <typename... T>
bool allHaveCpuEngines(const T&... tensor) {
  const std::vector<const Tensor*> tensorPtrs = {&tensor...};
  return tensorPtrs.empty() ||
      (hasCpuEngine(*tensorPtrs.front()) && haveSameEngines(tensorPtrs));
}

template <>
bool allHaveCpuEngines() {
  return true;
}

/**
 * ASSUME row-major layout. Compute the strides if we keep the dimensions but
 * permute the axes.
 */
dnnl::memory::dims getStridesAfterPermuteAxes(
    const dnnl::memory::dims& oldDims,
    const std::vector<Dim>& oldToNewAxes) {
  assert(oldDims.size() == oldToNewAxes.size());
  const auto ndim = oldDims.size();
  std::vector<Dim> newToOldAxes(ndim, 0);
  for (int oldAxis = 0; oldAxis < ndim; oldAxis++) {
    const auto newAxis = oldToNewAxes[oldAxis];
    newToOldAxes[newAxis] = oldAxis;
  }
  std::vector<dnnl::memory::dim> strides(ndim, 1);
  // calculate row major stride with new axes
  for (int newAxis = ndim - 2; newAxis >= 0; newAxis--) {
    const auto oldAxis = newToOldAxes[newAxis];
    const auto prevOldAxis = newToOldAxes[newAxis + 1];
    strides[oldAxis] = strides[prevOldAxis] * oldDims[prevOldAxis];
  }
  return dnnl::memory::dims(strides);
}

template <typename T, typename V>
Tensor fullWithTypeCpu(const Shape& shape, V value, const dtype type) {
  std::vector<T> data(shape.elements());
  std::fill(data.begin(), data.end(), static_cast<T>(value));
  return toTensor<OneDnnTensor>(shape, type, data.data(), Location::Host);
}

template <typename T, typename V>
Tensor fullWithType(const Shape& shape, V value, const dtype type) {
  const auto engineKind = OneDnnBackend::getInstance().engine().get_kind();
  if (engineKind == dnnl::engine::kind::cpu) {
    return fullWithTypeCpu<T, V>(shape, value, type);
  } else {
  throw std::runtime_error(
      "[OneDnnBackend::fullWithType] unimplemented for non-CPU engine");
  }
}

template <typename T>
Tensor iotaWithTypeCpu(const Shape& shape, const dtype type) {
  std::vector<T> data(shape.elements());
  std::iota(data.begin(), data.end(), 0);
  return toTensor<OneDnnTensor>(shape, type, data.data(), Location::Host);
}

Tensor iotaSingleAxisCpu(
    const unsigned ndims,
    const unsigned axis,
    const unsigned axisDim,
    const dtype type) {
  std::vector<Dim> dims(ndims, 1);
  dims[axis] = axisDim;
  const Shape shape(dims);
  switch (type) {
    case dtype::f16:
      return iotaWithTypeCpu<float>(shape, dtype::f32).astype(dtype::f16);
    case dtype::f32:
      return iotaWithTypeCpu<float>(shape, type);
    case dtype::f64:
      return iotaWithTypeCpu<double>(shape, type);
    case dtype::b8:
      return iotaWithTypeCpu<char>(shape, type);
    case dtype::s16:
      return iotaWithTypeCpu<short>(shape, type);
    case dtype::s32:
      return iotaWithTypeCpu<int>(shape, type);
    case dtype::s64:
      return iotaWithTypeCpu<long long>(shape, type);
    case dtype::u8:
      return iotaWithTypeCpu<unsigned char>(shape, type);
    case dtype::u16:
      return iotaWithTypeCpu<unsigned short>(shape, type);
    case dtype::u32:
      return iotaWithTypeCpu<unsigned int>(shape, type);
    case dtype::u64:
      return iotaWithTypeCpu<unsigned long long>(shape, type);
  }
}

struct BinaryOpOutputDesc {
  dnnl::memory::desc dstMemDesc;
  const Shape dstShape;
};

/**
 * Throw if not broadcastable(1) else select the right memory descriptor and
shape
 * for output.
 *
 * (1). Broadcast requires the following shapes
 *        LHS: (r1, ..., rn)
 *        RHS: (l1, ..., ln)
 *      where ri == li, or 1 ∈ (ri, li)
 *      output shape: (max(r1, l1), ..., max(rn, ln)
 *
 * TODO support broadcast with different # of dimensions. OneDNN's broadcast
 * requires inputs to have the same _number_ of dimensions, so we must either
 * "pad" the memory descriptor dimensions here or change our OneDnnTensor
 * representation to always pad to max dimension.
 */
BinaryOpOutputDesc getBinaryOpOutputDesc(
    const Shape& lhsShape,
    const dnnl::memory::desc& lhsMemDesc,
    const Shape& rhsShape,
    const dnnl::memory::desc& rhsMemDesc,
    std::optional<dnnl::memory::data_type> optDstType) {
  // allow implicit casting
  const auto dstType = optDstType.value_or(detail::getTypeWithLargerRange(
      lhsMemDesc.data_type(), rhsMemDesc.data_type()));
  // some common fast paths
  if (lhsShape == rhsShape) {
    return {
        .dstMemDesc =
            detail::oneDnnContiguousMemDescFromShape(lhsShape, dstType),
        .dstShape = lhsShape};
  }
  // only support same # of dimensions for now
  if (lhsShape.ndim() != rhsShape.ndim()) {
    std::stringstream ss;
    ss << "[OneDnnBackend] Cannot perform broadcast for tensors of shapes:"
       << lhsShape << " and " << rhsShape;
    throw std::runtime_error(ss.str());
  }
  // check and accumulate output dimensions
  auto ndim = lhsShape.ndim();
  std::vector<Dim> dstDims;
  for (auto i = 0; i < ndim; ++i) {
    auto lhsDim = lhsShape.get()[i];
    auto rhsDim = rhsShape.get()[i];
    if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1) {
      std::stringstream ss;
      ss << "[OneDnnBackend] Cannot perform broadcast for tensors of shapes:"
         << lhsShape << " and " << rhsShape;
      throw std::runtime_error(ss.str());
    }
    dstDims.push_back(std::max(lhsDim, rhsDim));
  }
  Shape dstShape(dstDims);
  return {
      .dstMemDesc = detail::oneDnnContiguousMemDescFromShape(dstShape, dstType),
      .dstShape = dstShape};
}

template <typename L, typename R, typename T, typename OP>
void applyBinopCpu(
    const void* lhs,
    const void* rhs,
    T* dst,
    unsigned count,
    OP op) {
  const L* lhsData = static_cast<const L*>(lhs);
  const R* rhsData = static_cast<const R*>(rhs);
  for (unsigned i = 0; i < count; i++) {
    dst[i] = op(lhsData[i], rhsData[i]);
  }
}

template <typename L, typename T, typename OP>
void applyBinopCpu(
    const void* lhs,
    const void* rhs,
    const dtype rhsType,
    T* dst,
    unsigned count,
    OP op) {
  switch (rhsType) {
    case fl::dtype::f16:
      throw std::runtime_error(
          "Fallback implementation currently doesn't support f16");
    case fl::dtype::f32:
      applyBinopCpu<L, float>(lhs, rhs, dst, count, op);
      break;
    case fl::dtype::f64:
      applyBinopCpu<L, double>(lhs, rhs, dst, count, op);
      break;
    case fl::dtype::b8:
      applyBinopCpu<L, char>(lhs, rhs, dst, count, op);
      break;
    case fl::dtype::s16:
      applyBinopCpu<L, short>(lhs, rhs, dst, count, op);
      break;
    case fl::dtype::s32:
      applyBinopCpu<L, int>(lhs, rhs, dst, count, op);
      break;
    case fl::dtype::s64:
      applyBinopCpu<L, long long>(lhs, rhs, dst, count, op);
      break;
    case fl::dtype::u8:
      applyBinopCpu<L, unsigned char>(lhs, rhs, dst, count, op);
      break;
    case fl::dtype::u16:
      applyBinopCpu<L, unsigned short>(lhs, rhs, dst, count, op);
      break;
    case fl::dtype::u32:
      applyBinopCpu<L, unsigned int>(lhs, rhs, dst, count, op);
      break;
    case fl::dtype::u64:
      applyBinopCpu<L, unsigned long long>(lhs, rhs, dst, count, op);
      break;
  }
}

template <typename T, typename OP>
void applyBinopCpu(
    const void* lhs,
    const dtype lhsType,
    const void* rhs,
    const dtype rhsType,
    T* dst,
    unsigned count,
    OP op) {
  switch (lhsType) {
    case fl::dtype::f16:
      throw std::runtime_error(
          "Fallback implementation currently doesn't support f16");
    case fl::dtype::f32:
      applyBinopCpu<float>(lhs, rhs, rhsType, dst, count, op);
      break;
    case fl::dtype::f64:
      applyBinopCpu<double>(lhs, rhs, rhsType, dst, count, op);
      break;
    case fl::dtype::b8:
      applyBinopCpu<char>(lhs, rhs, rhsType, dst, count, op);
      break;
    case fl::dtype::s16:
      applyBinopCpu<short>(lhs, rhs, rhsType, dst, count, op);
      break;
    case fl::dtype::s32:
      applyBinopCpu<int>(lhs, rhs, rhsType, dst, count, op);
      break;
    case fl::dtype::s64:
      applyBinopCpu<long long>(lhs, rhs, rhsType, dst, count, op);
      break;
    case fl::dtype::u8:
      applyBinopCpu<unsigned char>(lhs, rhs, rhsType, dst, count, op);
      break;
    case fl::dtype::u16:
      applyBinopCpu<unsigned short>(lhs, rhs, rhsType, dst, count, op);
      break;
    case fl::dtype::u32:
      applyBinopCpu<unsigned int>(lhs, rhs, rhsType, dst, count, op);
      break;
    case fl::dtype::u64:
      applyBinopCpu<unsigned long long>(lhs, rhs, rhsType, dst, count, op);
      break;
  }
}

template <typename T, typename OP>
Tensor sameShapeBinopCpu(const Tensor& lhs, const Tensor& rhs, OP op) {
  if (!lhs.isContiguous()) {
    return sameShapeBinopCpu<T>(lhs.asContiguousTensor(), rhs, op);
  }
  if (!rhs.isContiguous()) {
    return sameShapeBinopCpu<T>(lhs, rhs.asContiguousTensor(), op);
  }
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument(
        "[OneDnnBackend] Generic Binop impl requires input tensors of the same shape");
  }
  lhs.stream().sync();
  rhs.stream().sync();
  void* lhsData;
  void* rhsData;
  // On CPU, device pointer is host pointer.
  lhs.device(&lhsData);
  rhs.device(&rhsData);
  std::vector<T> dst(lhs.elements());
  T* dstData = dst.data();
  applyBinopCpu(
      lhsData, lhs.type(), rhsData, rhs.type(), dstData, lhs.elements(), op);
  lhs.unlock();
  rhs.unlock();
  auto dstType = dtype_traits<T>::fl_type;
  return toTensor<OneDnnTensor>(lhs.shape(), dstType, dstData, Location::Host);
}

template <typename T, typename OP>
Tensor sameShapeBinop(const Tensor& lhs, const Tensor& rhs, OP op) {
  if (hasCpuEngine(lhs) && hasCpuEngine(rhs)) {
    return sameShapeBinopCpu<T>(lhs, rhs, op);
  } else {
    throw std::runtime_error(
        "[OneDnnBackend::sameShapeBinop] unimplemented for non-CPU engine");
  }
}

Shape filterAxes(const Shape& shape, std::vector<int> axesToFilter) {
  std::vector<Dim> dimsKept;
  std::unordered_set<int> axesToFilterSet(
      axesToFilter.begin(), axesToFilter.end());
  for (int i = 0; i < shape.ndim(); i++) {
    if (axesToFilterSet.count(i) == 0) {
      dimsKept.push_back(shape[i]);
    }
  }
  return Shape(dimsKept);
}

template <typename INPUT_TYPE, typename CAST_TYPE>
Tensor createScalarTensorForBinop(const Tensor& tensor, INPUT_TYPE val) {
  CAST_TYPE castedVal = static_cast<CAST_TYPE>(val);
  Shape literalShape(std::vector<Dim>(tensor.ndim(), 1));
  auto type = dtype_traits<CAST_TYPE>::fl_type;
  return toTensor<OneDnnTensor>(
      literalShape, type, &castedVal, tensor.location());
}

dnnl::memory::desc transposeInnerMatrix(const dnnl::memory::desc& memDesc) {
  const auto ndims = memDesc.data.ndims;
  if (ndims < 2) {
    std::ostringstream oss;
    oss << "[transposeInnerMatrix] expected ndims to be >= 2, got: " << ndims;
    throw std::runtime_error(oss.str());
  }
  // recall that internal dims are reversed from the logical col-major dims
  std::vector<int> transposeAxesPermutation;
  for (int i = 0; i < ndims; i++) {
    transposeAxesPermutation.push_back(i);
  }
  std::swap(
      transposeAxesPermutation[ndims - 2], transposeAxesPermutation[ndims - 1]);
  return memDesc.permute_axes(transposeAxesPermutation);
}

std::tuple<Shape, Shape> padShorterDimsWithOnesOnTheRight(
    const Shape& tensorShape,
    const Shape& tileDims) {
  std::vector<Dim> paddedTensorDims = tensorShape.get();
  std::vector<Dim> paddedTileDims = tileDims.get();
  const auto tensorShapeNDims = tensorShape.ndim();
  const auto tileDimsNDims = tileDims.ndim();
  if (tensorShapeNDims > tileDimsNDims) {
    const auto diff = tensorShapeNDims - tileDimsNDims;
    paddedTileDims.insert(paddedTileDims.end(), diff, 1);
  } else {
    const auto diff = tileDimsNDims - tensorShapeNDims;
    paddedTensorDims.insert(paddedTensorDims.end(), diff, 1);
  }
  return {Shape(paddedTensorDims), Shape(paddedTileDims)};
}

} // namespace

OneDnnBackend::OneDnnBackend() {
#if FL_USE_MKL_RNG
  vslNewStream(&randStream_, VSL_BRNG_MCG31, std::rand());
#else
  randEngine_ = RandEngineType{static_cast<uint_fast32_t>(std::rand())};
#endif // FL_USE_MKL_RNG
  engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
  stream_ = OneDnnCPUStream::create(engine_);
}

OneDnnBackend& OneDnnBackend::getInstance() {
  static OneDnnBackend instance;
  return instance;
}

TensorBackendType OneDnnBackend::backendType() const {
  return TensorBackendType::OneDnn;
}

const Stream& OneDnnBackend::stream() const {
  return *stream_;
}

dnnl::stream& OneDnnBackend::nativeStream() const {
  return stream_->handle();
}

const dnnl::engine& OneDnnBackend::engine() const {
  return engine_;
}

const dnnl::engine& OneDnnBackend::cpuEngine() const {
  return engine_;
}

/* -------------------------- Compute Functions -------------------------- */

void OneDnnBackend::eval(const Tensor& /* tensor */) {
  // no-op since OneDNN computations are launched eagerly
}

bool OneDnnBackend::supportsDataType(const fl::dtype& type) const {
  return detail::isTypeSupportedByOneDnn(type);
}

void OneDnnBackend::getMemMgrInfo(
    const char* /* msg */,
    const int /* deviceId */,
    std::ostream* /* ostream */) {
  throw std::runtime_error(
      "[OneDnnBackend] Currently no memory manager support");
}

void OneDnnBackend::setMemMgrLogStream(std::ostream* /* stream */) {
  throw std::runtime_error(
      "[OneDnnBackend] Currently no memory manager support");
}

void OneDnnBackend::setMemMgrLoggingEnabled(const bool /* enabled */) {
  throw std::runtime_error(
      "[OneDnnBackend] Currently no memory manager support");
}

void OneDnnBackend::setMemMgrFlushInterval(const size_t /* interval */) {
  throw std::runtime_error(
      "[OneDnnBackend] Currently no memory manager support");
}

/* -------------------------- Rand Functions -------------------------- */

void OneDnnBackend::setSeed(const int seed) {
#if FL_USE_MKL_RNG
  vslDeleteStream(&randStream_);
  vslNewStream(&randStream_, VSL_BRNG_MCG31, seed);
#else
  randEngine_.seed(seed);
#endif // FL_USE_MKL_RNG
}

Tensor OneDnnBackend::randnCpu(const Shape& shape, const dtype type) {
  std::vector<float> data(shape.elements());
#if FL_USE_MKL_RNG
  const auto alg = VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2;
  vsRngGaussian(alg, randStream_, data.size(), data.data(), 0, 1);
#else
  std::normal_distribution<float> normal_dist(0, 1);
  auto* data_ptr = data.data();
  for (decltype(data.size()) i = 0; i < data.size(); ++i) {
    data_ptr[i] = normal_dist(randEngine_);
  }
#endif // FL_USE_MKL_RNG
  return toTensor<OneDnnTensor>(shape, dtype::f32, data.data(), Location::Host)
      .astype(type);
}

Tensor OneDnnBackend::randCpu(const Shape& shape, const dtype type) {
  std::vector<float> data(shape.elements());
#if FL_USE_MKL_RNG
  const auto alg = VSL_RNG_METHOD_UNIFORM_STD;
  vsRngUniform(alg, randStream_, data.size(), data.data(), 0, 1);
#else
  std::uniform_real_distribution<float> uniform_dist{};
  auto* data_ptr = data.data();
  for (decltype(data.size()) i = 0; i < data.size(); ++i) {
    data_ptr[i] = uniform_dist(randEngine_);
  }
#endif // FL_USE_MKL_RNG
  return toTensor<OneDnnTensor>(shape, dtype::f32, data.data(), Location::Host)
      .astype(type);
}

Tensor OneDnnBackend::randn(const Shape& shape, dtype type) {
  if (engine_.get_kind() == dnnl::engine::kind::cpu) {
    return randnCpu(shape, type);
  } else {
    throw std::runtime_error(
        "[OneDnnBackend::randn] unimplemented for non-CPU engine");
  }
}

Tensor OneDnnBackend::rand(const Shape& shape, dtype type) {
  if (engine_.get_kind() == dnnl::engine::kind::cpu) {
    return randCpu(shape, type);
  } else {
    throw std::runtime_error(
        "[OneDnnBackend::rand] unimplemented for non-CPU engine");
  }
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
#define FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(TYPE)                         \
  Tensor OneDnnBackend::fromScalar(TYPE /* value */, const dtype /* type */) { \
    throw std::invalid_argument(                                               \
        "OneDnnBackend::fromScalar - not implemented for type " +              \
        std::string(#TYPE));                                                   \
  }                                                                            \
  Tensor OneDnnBackend::full(                                                  \
      const Shape& shape, TYPE value, const dtype type) {                      \
    switch (type) {                                                            \
      case dtype::f16:                                                         \
        return fullWithType<float>(shape, value, dtype::f32)                   \
            .astype(dtype::f16);                                               \
      case dtype::f32:                                                         \
        return fullWithType<float>(shape, value, type);                        \
      case dtype::f64:                                                         \
        return fullWithType<double>(shape, value, type);                       \
      case dtype::b8:                                                          \
        return fullWithType<char>(shape, value, type);                         \
      case dtype::s16:                                                         \
        return fullWithType<short>(shape, value, type);                        \
      case dtype::s32:                                                         \
        return fullWithType<int>(shape, value, type);                          \
      case dtype::s64:                                                         \
        return fullWithType<long long>(shape, value, type);                    \
      case dtype::u8:                                                          \
        return fullWithType<unsigned char>(shape, value, type);                \
      case dtype::u16:                                                         \
        return fullWithType<unsigned short>(shape, value, type);               \
      case dtype::u32:                                                         \
        return fullWithType<unsigned int>(shape, value, type);                 \
      case dtype::u64:                                                         \
        return fullWithType<unsigned long long>(shape, value, type);           \
    }                                                                          \
  }
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const double&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const float&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const int&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const char&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned char&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const long&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned long&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const long long&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned long long&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const bool&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const short&);
FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned short&);
#undef FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DEF

template <typename T, typename V>
Tensor OneDnnBackend::fullWithType(const Shape& shape, V value, const dtype type) {
  if (engine_.get_kind() == dnnl::engine::kind::cpu) {
    return fullWithTypeCpu<T, V>(shape, value, type);
  } else {
  throw std::runtime_error(
      "[OneDnnBackend::fullWithType] unimplemented for non-CPU engine");
  }
}

Tensor OneDnnBackend::identity(const Dim /* dim */, const dtype /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::arange(
    const Shape& shape,
    const Dim seqDim,
    const dtype type) {
  if (seqDim < 0 || seqDim >= shape.ndim()) {
    std::ostringstream oss;
    oss << "[OneDnnBackend::arange] Invalid seqDim: " << seqDim
        << ", for shape: " << shape;
    throw std::invalid_argument(oss.str());
  }
  if (engine_.get_kind() == dnnl::engine::kind::cpu) {
    std::vector<Dim> tileDims = shape.get();
    tileDims[seqDim] = 1;
    return tile(
        iotaSingleAxisCpu(shape.ndim(), seqDim, shape[seqDim], type),
        Shape(tileDims));
  } else {
    throw std::runtime_error(
        "[OneDnnBackend::arange] unimplemented for non-CPU engine");
  }
}

Tensor OneDnnBackend::iota(
    const Shape& /* dims */,
    const Shape& /* tileDims */,
    const dtype /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

/************************ Shaping and Indexing *************************/
Tensor OneDnnBackend::reshape(
    const Tensor& tensor,
    const Shape& shape) {
  if (tensor.shape().elements() != shape.elements()) {
    std::ostringstream oss;
    oss << "[OneDnnBackend::reshape] Cannot reshape tensor from "
        << tensor.shape() << " to " << shape;
    throw std::invalid_argument(oss.str());
  }

  // TODO copy on write.
  // prepare memories
  auto& srcTensor = toOneDnnTensor(tensor);
  auto& mem = srcTensor.memory();
  const auto& memDesc = srcTensor.memoryDesc();
  const auto reshapedMemDesc =
      detail::oneDnnContiguousMemDescFromShape(shape, memDesc.data_type());
  auto reshapedMem = dnnl::memory(reshapedMemDesc, engine_);

  // prepare primitive (use reorder to do a copy)
  const auto reorderPrimitiveDesc = dnnl::reorder::primitive_desc(
      engine_, memDesc, engine_, memDesc);
  const auto reorderPrimitive = dnnl::reorder(reorderPrimitiveDesc);

  // execute primitive
  reorderPrimitive.execute(stream_->handle(), mem, reshapedMem);
  return toTensor<OneDnnTensor>(shape, std::move(reshapedMem));
}

// 1. OneDNN doesn't have native support for tensor transpose.
// 2. `reorder` is the best primitive to move data in this case.
// 3. `reorder` requires same dims for input & output.
// 4. Our final output memory needs to have dims transposed.
//
// Due to the limitations above, this is what we'll do:
//   0. create output memory with dims transposed.
//   1. reorder memory based on a new output memory descriptor (similar to a
//   view) where we use input dims and the transposed layout (specified as
//   strides due to API limitation)
//
// Logically, the relationship among dnnl::memory transformations is as follows:
//         [[1 2 3],
//          [4 5 6]]
//             |     \
// (transpose) |      \
//             v       \
//          [[1 4],     |
//           [2 5],     | (reorder)
//           [3 6]]     |
//             ^        |
//   (reshape) |        /
//             v      /
//         [[1 4 2], <
//          [5 3 6]]
//
// In other words, we are simulating transpose via reorder & reshape.
Tensor OneDnnBackend::transpose(
    const Tensor& tensor,
    const Shape& axes /* = {} */) {
  if (tensor.ndim() <= 1) {
    return tensor.copy();
  }
  Shape newShape = tensor.shape();
  std::vector<Dim> oldToNewAxes = axes.get();
  if (axes.ndim() == 0) { // default, reverse all axes
    oldToNewAxes.resize(tensor.ndim());
    std::reverse(newShape.get().begin(), newShape.get().end());
    std::iota(oldToNewAxes.begin(), oldToNewAxes.end(), 0);
    std::reverse(oldToNewAxes.begin(), oldToNewAxes.end());
  } else if (axes.ndim() == tensor.ndim()) {
    for (int axis = 0; axis < axes.ndim(); axis++) {
      newShape[axis] = tensor.dim(oldToNewAxes[axis]);
    }
  } else {
    std::invalid_argument(
        "[OneDnnBackend::transpose] Invalid axes: " + axes.toString() +
        " for shape: " + tensor.shape().toString());
  }

  // prepare memories
  auto& srcTensor = toOneDnnTensor(tensor);
  auto& srcMem = srcTensor.memory();
  const auto& srcMemDesc = srcTensor.memoryDesc();
  const auto type = srcMemDesc.data_type();
  const auto srcMemDims = srcMemDesc.dims();
  const auto dstMemDesc =
      detail::oneDnnContiguousMemDescFromShape(newShape, type);
  auto dstMem = dnnl::memory(dstMemDesc, engine_);

  // prepare primitive
  const auto reorderDstStrides =
      getStridesAfterPermuteAxes(srcMemDims, oldToNewAxes);
  const auto reorderDstMemDesc =
      dnnl::memory::desc(srcMemDims, type, reorderDstStrides);
  const auto reorderPrimitiveDesc = dnnl::reorder::primitive_desc(
      engine_, srcMemDesc, engine_, reorderDstMemDesc);
  const auto reorderPrimitive = dnnl::reorder(reorderPrimitiveDesc);

  // execute primitive
  reorderPrimitive.execute(stream_->handle(), srcMem, dstMem);
  return toTensor<OneDnnTensor>(newShape, std::move(dstMem));
}

Tensor OneDnnBackend::tile(
    const Tensor& tensor,
    const Shape& tileDims) {
  const auto [paddedTensorShape, paddedTileDims] =
    padShorterDimsWithOnesOnTheRight(tensor.shape(), tileDims);

  auto& srcTensor = toOneDnnTensor(tensor);
  auto currTiledMem = srcTensor.memory();
  auto currTiledMemDesc =
    srcTensor.memoryDesc().reshape(detail::shapeToOneDnnDims(paddedTensorShape));
  std::vector<Dim> finalDims;
  // TODO use uniform axes once we remove the 'transposed' representation
  for (int shapeAxis = 0; shapeAxis < paddedTileDims.ndim(); shapeAxis++) {
    const auto dimsAxis = paddedTensorShape.ndim() - 1 - shapeAxis;
    const auto numTiles = paddedTileDims[shapeAxis];
    finalDims.push_back(numTiles * paddedTensorShape[shapeAxis]);
    if (numTiles > 1) {
      // prepare memories
      std::vector<dnnl::memory::desc> tileMemDescs(numTiles, currTiledMemDesc);

      // prepare concat primitive
      const dnnl::concat::primitive_desc concatPrimitiveDesc(
          dimsAxis, tileMemDescs, engine_);
      const dnnl::concat concatPrimitive(concatPrimitiveDesc);
      const auto newTileMemDesc = concatPrimitiveDesc.dst_desc();
      auto newTiledMem = dnnl::memory(newTileMemDesc, engine_);

      // prepare arguments.
      std::unordered_map<int, dnnl::memory> args{{DNNL_ARG_DST, newTiledMem}};
      for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
        args.insert({DNNL_ARG_MULTIPLE_SRC + tileIdx, currTiledMem});
      }

      // execute primitive
      concatPrimitive.execute(stream_->handle(), args);
      currTiledMemDesc = newTileMemDesc;
      currTiledMem = newTiledMem;
    }
  }
  return toTensor<OneDnnTensor>(Shape(finalDims), std::move(currTiledMem));
}

Tensor OneDnnBackend::concatenate(
    const std::vector<Tensor>& /* tensors */,
    const unsigned /* axis */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::nonzero(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::pad(
    const Tensor& /* input */,
    const std::vector<std::pair<int, int>>& /* padWidths */,
    const PadType /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

/************************** Unary Operators ***************************/

Tensor OneDnnBackend::exp(const Tensor& tensor) {
  return applyEltwiseOp(tensor, dnnl::algorithm::eltwise_exp);
}

Tensor OneDnnBackend::log(const Tensor& tensor) {
  return applyEltwiseOp(tensor, dnnl::algorithm::eltwise_log);
}

Tensor OneDnnBackend::negative(const Tensor& tensor) {
  return applyEltwiseOp(tensor, dnnl::algorithm::eltwise_linear, -1);
}

Tensor OneDnnBackend::logicalNot(const Tensor& tensor) {
  return tensor == 0;
}

Tensor OneDnnBackend::log1p(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::sin(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::cos(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::sqrt(const Tensor& tensor) {
  return applyEltwiseOp(tensor, dnnl::algorithm::eltwise_sqrt);
}

Tensor OneDnnBackend::tanh(const Tensor& tensor) {
  return applyEltwiseOp(tensor, dnnl::algorithm::eltwise_tanh);
}

Tensor OneDnnBackend::floor(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::ceil(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::rint(const Tensor& tensor) {
  return applyEltwiseOp(tensor, dnnl::algorithm::eltwise_round);
}

Tensor OneDnnBackend::absolute(const Tensor& tensor) {
  return applyEltwiseOp(tensor, dnnl::algorithm::eltwise_abs);
}

Tensor OneDnnBackend::sigmoid(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::erf(const Tensor& tensor) {
  // gelu_erf is the only OneDNN primitive that contains erf, basically
  // gelu_erf(a) = 0.5a(1 + erf(a/sqrt(2)))
  auto sqrt2 = std::sqrt(2);
  auto tensorSqrt2 = tensor * sqrt2;
  auto res = applyEltwiseOp(tensorSqrt2, dnnl::algorithm::eltwise_gelu_erf);
  return res / (tensorSqrt2 * 0.5) - 1;
  // TODO investigate performance using post-ops -- just launch 1 primitive here
}

Tensor OneDnnBackend::flip(
    const Tensor& /* tensor */,
    const unsigned /* dim */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::clip(
    const Tensor& /* tensor */,
    const Tensor& /* low */,
    const Tensor& /* high */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::clip(
    const Tensor& tensor,
    const double& low,
    const double& high) {
  return applyEltwiseOp(tensor, dnnl::algorithm::eltwise_clip, low, high);
}

Tensor OneDnnBackend::roll(
    const Tensor& /* tensor */,
    const int /* shift */,
    const unsigned /* axis */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::isnan(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::isinf(const Tensor& tensor) {
  return tensor == std::numeric_limits<float>::infinity() ||
      tensor == (-1 * std::numeric_limits<float>::infinity());
}

Tensor OneDnnBackend::sign(const Tensor& tensor) {
  return (0 < tensor) - (tensor < 0);
}

Tensor OneDnnBackend::tril(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::triu(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::where(
    const Tensor& /* condition */,
    const Tensor& /* x */,
    const Tensor& /* y */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

void OneDnnBackend::topk(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* k */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::sort(
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

void OneDnnBackend::sort(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::argsort(
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::applyEltwiseOp(
    const Tensor& tensor,
    const dnnl::algorithm alg,
    float alpha /* 0 */,
    float beta /* 0 */) {
  // prepare memories
  auto& srcTensor = toOneDnnTensor(tensor);
  const auto mem = srcTensor.memory();
  const auto& memDesc = srcTensor.memoryDesc();
  const auto dstMemDesc = detail::oneDnnContiguousMemDescFromShape(
      tensor.shape(), memDesc.data_type());
  auto dstMem = dnnl::memory(dstMemDesc, engine_);

  // prepare unary primitive
  const auto unaryDesc = dnnl::eltwise_forward::desc(
      dnnl::prop_kind::forward_inference, alg, memDesc, alpha, beta);
  const auto unaryPrimtiveDesc =
      dnnl::eltwise_forward::primitive_desc(unaryDesc, engine_);
  const auto unaryPrimitive = dnnl::eltwise_forward(unaryPrimtiveDesc);

  // prepare arguments.
  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, mem},
      {DNNL_ARG_DST, dstMem},
  };

  // execute primitive
  unaryPrimitive.execute(stream_->handle(), args);
  return toTensor<OneDnnTensor>(tensor.shape(), std::move(dstMem));
}

/************************** Binary Operators ***************************/
#define FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(FUNC, OP, TYPE)              \
  Tensor OneDnnBackend::FUNC(const Tensor& /* a */, TYPE /* rhs */) {         \
    throw std::runtime_error(                                                 \
        "OneDnnBackend::" + std::string(#FUNC) + " unimplemented for type " + \
        std::string(#TYPE));                                                  \
  }                                                                           \
  Tensor OneDnnBackend::FUNC(TYPE /* lhs */, const Tensor& /* a */) {         \
    throw std::runtime_error(                                                 \
        "OneDnnBackend::" + std::string(#FUNC) + " unimplemented for type " + \
        std::string(#TYPE));                                                  \
  }

#define FL_ONEDNN_BINARY_OP_LITERALS_UNSUPPORTED_DEF(FUNC, OP)              \
  FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(FUNC, OP, const bool&);          \
  FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(FUNC, OP, const int&);           \
  FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(FUNC, OP, const unsigned&);      \
  FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(FUNC, OP, const char&);          \
  FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(FUNC, OP, const unsigned char&); \
  FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(FUNC, OP, const long&);          \
  FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(FUNC, OP, const unsigned long&); \
  FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(FUNC, OP, const long long&);     \
  FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(                                 \
      FUNC, OP, const unsigned long long&);                                 \
  FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(FUNC, OP, const double&);        \
  FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(FUNC, OP, const float&);         \
  FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(FUNC, OP, const short&);         \
  FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF(FUNC, OP, const unsigned short&);

// Operations on fl::Tensor call the respective operator overloads that are
// already defined on af::arrays
#define FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(OP, FUNC)     \
  Tensor OneDnnBackend::FUNC(                             \
      const Tensor& /* lhs */, const Tensor& /* rhs */) { \
    throw std::runtime_error(                             \
        "OneDnnBackend::" + std::string(#FUNC) +          \
        " unimplemented for two-Tensor inputs.");         \
  }                                                       \
  FL_ONEDNN_BINARY_OP_LITERALS_UNSUPPORTED_DEF(FUNC, OP);

FL_ONEDNN_BINARY_OP_LITERALS_UNSUPPORTED_DEF(logicalOr, ||);
FL_ONEDNN_BINARY_OP_LITERALS_UNSUPPORTED_DEF(logicalAnd, &&);
FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(%, mod);
FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(&, bitwiseAnd);
FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(|, bitwiseOr);
FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(^, bitwiseXor);
FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(<<, lShift);
FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(>>, rShift);
#undef FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF
#undef FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF
#undef FL_ONEDNN_BINARY_OP_LITERALS_UNSUPPORTED_DEF

#define FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, TYPE, CAST_TYPE)          \
  Tensor OneDnnBackend::FUNC(const Tensor& a, TYPE rhs) {                \
    return FUNC(a, createScalarTensorForBinop<TYPE, CAST_TYPE>(a, rhs)); \
  }                                                                      \
  Tensor OneDnnBackend::FUNC(TYPE lhs, const Tensor& a) {                \
    return FUNC(createScalarTensorForBinop<TYPE, CAST_TYPE>(a, lhs), a); \
  }

// NOTE cast needed because OneDNN only supports a subset of the FL types.
#define FL_BINARY_OP_LITERALS_DEF(FUNC)                                     \
  FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, const bool&, float);               \
  FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, const int&, int);                  \
  FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, const unsigned&, unsigned);        \
  FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, const char&, float);               \
  FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, const unsigned char&, float);      \
  FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, const long&, int);                 \
  FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, const unsigned long&, unsigned);   \
  FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, const long long&, float);          \
  FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, const unsigned long long&, float); \
  FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, const double&, float);             \
  FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, const float&, float);              \
  FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, const short&, float);              \
  FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, const unsigned short&, float);

#define FL_ONEDNN_BINARY_OP_DEF(FUNC, ALG)                           \
  Tensor OneDnnBackend::FUNC(const Tensor& lhs, const Tensor& rhs) { \
    return applyBinop(lhs, rhs, ALG);                                \
  }                                                                  \
  FL_BINARY_OP_LITERALS_DEF(FUNC);

#define FL_ONEDNN_BINARY_LOGICAL_OP_DEF(FUNC, ALG)                   \
  Tensor OneDnnBackend::FUNC(const Tensor& lhs, const Tensor& rhs) { \
    return applyBinop(lhs, rhs, ALG, dnnl::memory::data_type::s8);   \
  }                                                                  \
  FL_BINARY_OP_LITERALS_DEF(FUNC);

FL_ONEDNN_BINARY_OP_DEF(add, dnnl::algorithm::binary_add);
FL_ONEDNN_BINARY_OP_DEF(sub, dnnl::algorithm::binary_sub);
FL_ONEDNN_BINARY_OP_DEF(mul, dnnl::algorithm::binary_mul);
FL_ONEDNN_BINARY_OP_DEF(div, dnnl::algorithm::binary_div);
FL_ONEDNN_BINARY_LOGICAL_OP_DEF(eq, dnnl::algorithm::binary_eq);
FL_ONEDNN_BINARY_LOGICAL_OP_DEF(neq, dnnl::algorithm::binary_ne);
FL_ONEDNN_BINARY_LOGICAL_OP_DEF(lessThan, dnnl::algorithm::binary_lt);
FL_ONEDNN_BINARY_LOGICAL_OP_DEF(lessThanEqual, dnnl::algorithm::binary_le);
FL_ONEDNN_BINARY_LOGICAL_OP_DEF(greaterThan, dnnl::algorithm::binary_gt);
FL_ONEDNN_BINARY_LOGICAL_OP_DEF(greaterThanEqual, dnnl::algorithm::binary_ge);
#undef FL_ONEDNN_BINARY_OP_DEF

Tensor OneDnnBackend::logicalAnd(const Tensor& lhs, const Tensor& rhs) {
  return sameShapeBinop<char>(lhs, rhs, std::logical_and<>());
}

Tensor OneDnnBackend::logicalOr(const Tensor& lhs, const Tensor& rhs) {
  return sameShapeBinop<char>(lhs, rhs, std::logical_or<>());
}

Tensor OneDnnBackend::minimum(const Tensor& lhs, const Tensor& rhs) {
  return applyBinop(lhs, rhs, dnnl::algorithm::binary_min);
}

Tensor OneDnnBackend::minimum(const Tensor& lhs, const double& rhs) {
  return minimum(lhs, createScalarTensorForBinop<double, float>(lhs, rhs));
}

Tensor OneDnnBackend::minimum(const double& lhs, const Tensor& rhs) {
  return minimum(rhs, lhs); // commutative
}

Tensor OneDnnBackend::maximum(const Tensor& lhs, const Tensor& rhs) {
  return applyBinop(lhs, rhs, dnnl::algorithm::binary_max);
}

Tensor OneDnnBackend::maximum(const Tensor& lhs, const double& rhs) {
  return maximum(lhs, createScalarTensorForBinop<double, float>(lhs, rhs));
}

Tensor OneDnnBackend::maximum(const double& lhs, const Tensor& rhs) {
  return maximum(rhs, lhs); // commutative
}

Tensor OneDnnBackend::applyBinop(
    const Tensor& lhs,
    const Tensor& rhs,
    dnnl::algorithm alg,
    std::optional<dnnl::memory::data_type> dstType /* = std::nullopt */) {
  // prepare memories
  auto& lhsTensor = toOneDnnTensor(lhs);
  auto& rhsTensor = toOneDnnTensor(rhs);
  auto lhsMem = lhsTensor.memory();
  auto rhsMem = rhsTensor.memory();
  const auto& lhsMemDesc = lhsTensor.memoryDesc();
  const auto& rhsMemDesc = rhsTensor.memoryDesc();
  const auto outputDesc = getBinaryOpOutputDesc(
      lhs.shape(), lhsMemDesc, rhs.shape(), rhsMemDesc, dstType);
  auto dstMem = dnnl::memory(outputDesc.dstMemDesc, engine_);

  // prepare primitive
  const auto binaryDesc = dnnl::binary::desc(
    alg, lhsMemDesc, rhsMemDesc, outputDesc.dstMemDesc);
  const auto binaryPrimtiveDesc =
      dnnl::binary::primitive_desc(binaryDesc, engine_);
  const auto binaryPrimitive = dnnl::binary(binaryPrimtiveDesc);

  // prepare arguments
  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC_0, lhsMem},
      {DNNL_ARG_SRC_1, rhsMem},
      {DNNL_ARG_DST, dstMem},
  };

  // execute primitive
  binaryPrimitive.execute(stream_->handle(), args);
  return toTensor<OneDnnTensor>(outputDesc.dstShape, std::move(dstMem));
}

Tensor OneDnnBackend::power(const Tensor& /* lhs */, const Tensor& /* rhs */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::power(const Tensor& lhs, const double& rhs) {
  // alpha * element^beta
  return applyEltwiseOp(lhs, dnnl::algorithm::eltwise_pow, 1, rhs);
}

/************************** BLAS ***************************/

Tensor OneDnnBackend::matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp) {
  std::vector<Dim> lhsDims = lhs.shape().get();
  std::vector<Dim> rhsDims = rhs.shape().get();
  const bool isLhsScalarOrVector = lhsDims.size() <= 1;
  const bool isRhsScalarOrVector = rhsDims.size() <= 1;
  auto& lhsTensor = toOneDnnTensor(lhs);
  auto& rhsTensor = toOneDnnTensor(rhs);
  auto lhsMem = lhsTensor.memory();
  auto rhsMem = rhsTensor.memory();
  auto lhsMemDesc = lhsTensor.memoryDesc();
  auto rhsMemDesc = rhsTensor.memoryDesc();
  if (isLhsScalarOrVector) { // pad to (1 x 1/K)
    lhsDims.insert(lhsDims.end(), 2 - lhsDims.size(), 1);
    std::reverse(lhsDims.begin(), lhsDims.end());
    lhsMemDesc = lhsMemDesc.reshape(detail::flDimsToOneDnnDims(lhsDims));
  } else if (lhsProp == MatrixProperty::Transpose) {
    std::swap(lhsDims[0], lhsDims[1]);
    lhsMemDesc = transposeInnerMatrix(lhsMemDesc);
  }
  if (isRhsScalarOrVector) { // pad to (1/K x 1)
    rhsDims.insert(rhsDims.end(), 2 - rhsDims.size(), 1);
    rhsMemDesc = rhsMemDesc.reshape(detail::flDimsToOneDnnDims(rhsDims));
  } else if (rhsProp == MatrixProperty::Transpose) {
    std::swap(rhsDims[0], rhsDims[1]);
    rhsMemDesc = transposeInnerMatrix(rhsMemDesc);
  }

  // shape check (TODO support broadcasting)
  if (!(lhsDims.at(1) == rhsDims.at(0) &&
        std::equal(
          lhsDims.begin() + 2,
          lhsDims.end(),
          rhsDims.begin() + 2,
          rhsDims.end()))) {
    std::ostringstream oss;
    oss << "Cannot perform matmul for tensors of shapes: " << lhs.shape()
      << " and " << rhs.shape();
    throw std::invalid_argument(oss.str());
  }
  std::vector<Dim> dstDims = lhsDims;
  dstDims[1] = rhsDims[1];
  Shape dstShape(dstDims);

  // prepare memories
  const auto dstType = detail::getTypeWithLargerRange(
      lhsMemDesc.data_type(), rhsMemDesc.data_type());
  const auto dstMemArgDesc =
      detail::oneDnnContiguousMemDescFromShape(dstShape, dstType);
  auto dstMemDesc = dstMemArgDesc;
  // For such cases, keep output as a vector instead of 2d matrix,
  // but the matmul primitive requries the 2d dims, thus dstMemArgDesc.
  if (isLhsScalarOrVector || isRhsScalarOrVector) {
    const auto elems = dstShape.elements();
    dstMemDesc = dstMemArgDesc.reshape({elems});
    dstShape = {elems};
  }
  auto dstMem = dnnl::memory(dstMemDesc, engine_);

  // NOTE since our physical representation is a transpose of the logical
  // representation, we must switch lhs/rhs during matmul. i.e.,
  //  logical      physical
  //     A            AT
  //     B            BT
  //   A x B        (A x B)T = BT x AT
  // TODO once we support arbitrary internal layout, we can get rid of this.
  const auto& srcMemDesc = rhsMemDesc;
  auto& srcMem = rhsMem;
  const auto& weightsMemDesc = lhsMemDesc;
  auto& weightsMem = lhsMem;

  // prepare primitive
  const auto matmulDesc =
    dnnl::matmul::desc(srcMemDesc, weightsMemDesc, dstMemArgDesc);
  const auto matmulPrimitiveDesc =
    dnnl::matmul::primitive_desc(matmulDesc, engine_);
  const auto matmulPrimitive = dnnl::matmul(matmulPrimitiveDesc);

  // prepare arguments.
  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, srcMem},
      {DNNL_ARG_WEIGHTS, weightsMem},
      {DNNL_ARG_DST, dstMem},
  };

  // execute primitive
  matmulPrimitive.execute(stream_->handle(), args);
  return toTensor<OneDnnTensor>(dstShape, std::move(dstMem));
}

/************************** Reductions ***************************/

Tensor OneDnnBackend::amin(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  return applyReductionOp(
      input, dnnl::algorithm::reduction_min, axes, keepDims);
}

Tensor OneDnnBackend::amax(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  return applyReductionOp(
      input, dnnl::algorithm::reduction_max, axes, keepDims);
}

template <typename T, typename LessThan>
void maxWithIndexCpu(
    Tensor& values,
    Tensor& indices,
    const Shape& inputShape,
    const std::vector<T>& inputData,
    const unsigned axis,
    const bool keepDims,
    LessThan lt) {
  const auto inputElemCount = inputShape.elements();
  const auto axisSize = inputShape.dim(axis);
  const unsigned axisStride = std::accumulate(
      inputShape.get().begin(),
      inputShape.get().begin() + axis,
      1,
      std::multiplies<>());
  const auto resultElemCount = inputElemCount / axisSize;
  std::vector<T> valuesData;
  std::vector<int> indicesData;
  valuesData.reserve(resultElemCount);
  indicesData.reserve(resultElemCount);
  // default to falses:
  auto visited = std::make_unique<bool[]>(inputElemCount);
  for (unsigned idx = 0; idx < inputElemCount; idx++) {
    if (!visited[idx]) {
      visited[idx] = true;
      T maxVal = inputData[idx];
      int maxAxisIdx = 0;
      // search for max element & index along this unvisted axis
      for (unsigned axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        int elemIdx = idx + axisIdx * axisStride;
        assert(!visited[elemIdx]);
        visited[elemIdx] = true;
        T elemVal = inputData[elemIdx];
        if (lt(maxVal, elemVal)) {
          maxVal = elemVal;
          maxAxisIdx = axisIdx;
        }
      }
      valuesData.push_back(maxVal);
      indicesData.push_back(maxAxisIdx);
    }
  }
  // write data to output tensors
  std::vector<Dim> outputDims = inputShape.get();
  if (keepDims) {
    outputDims[axis] = 1;
  } else {
    outputDims.erase(outputDims.begin() + axis);
  }
  auto valType = dtype_traits<T>::fl_type;
  values = toTensor<OneDnnTensor>(
      Shape(outputDims), valType, valuesData.data(), Location::Host);
  indices = toTensor<OneDnnTensor>(
      Shape(outputDims), dtype::s32, indicesData.data(), Location::Host);
}

template <typename LessThan>
void maxWithIndexCpu(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims,
    LessThan lt) {
  if (axis >= input.ndim()) {
    std::ostringstream oss;
    oss << "[OneDnnBackend::min] Axis too large: " << axis
        << " for tensor of shape: " << input.shape();
    throw std::invalid_argument(oss.str());
  }
  const Shape& inputShape = input.shape();
  switch (input.type()) {
    case dtype::f16:
      throw std::runtime_error("[OneDnnTensor::min] doesn't support f16");
    case dtype::f32: {
      auto dataVec = input.toHostVector<float>();
      maxWithIndexCpu(values, indices, inputShape, dataVec, axis, keepDims, lt);
      return;
    }
    case dtype::f64: {
      auto dataVec = input.toHostVector<double>();
      maxWithIndexCpu(values, indices, inputShape, dataVec, axis, keepDims, lt);
      return;
    }
    case dtype::b8: {
      auto dataVec = input.toHostVector<char>();
      maxWithIndexCpu(values, indices, inputShape, dataVec, axis, keepDims, lt);
      return;
    }
    case dtype::s16: {
      auto dataVec = input.toHostVector<short>();
      maxWithIndexCpu(values, indices, inputShape, dataVec, axis, keepDims, lt);
      return;
    }
    case dtype::s32: {
      auto dataVec = input.toHostVector<int>();
      maxWithIndexCpu(values, indices, inputShape, dataVec, axis, keepDims, lt);
      return;
    }
    case dtype::s64: {
      auto dataVec = input.toHostVector<long long>();
      maxWithIndexCpu(values, indices, inputShape, dataVec, axis, keepDims, lt);
      return;
    }
    case dtype::u8: {
      auto dataVec = input.toHostVector<unsigned char>();
      maxWithIndexCpu(values, indices, inputShape, dataVec, axis, keepDims, lt);
      return;
    }
    case dtype::u16: {
      auto dataVec = input.toHostVector<unsigned short>();
      maxWithIndexCpu(values, indices, inputShape, dataVec, axis, keepDims, lt);
      return;
    }
    case dtype::u32: {
      auto dataVec = input.toHostVector<unsigned int>();
      maxWithIndexCpu(values, indices, inputShape, dataVec, axis, keepDims, lt);
      return;
    }
    case dtype::u64: {
      auto dataVec = input.toHostVector<unsigned long long>();
      maxWithIndexCpu(values, indices, inputShape, dataVec, axis, keepDims, lt);
      return;
    }
  }
}

// TODO move this into a generic CPU backend
void OneDnnBackend::min(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  if (allHaveCpuEngines(values, indices, input)) {
    return maxWithIndexCpu(
        values, indices, input, axis, keepDims, std::greater<>());
  } else {
    throw std::runtime_error(
        "[OneDnnBackend::min] unimplemented for non-CPU engine");
  }
}

// TODO move this into a generic CPU backend
void OneDnnBackend::max(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  if (allHaveCpuEngines(values, indices, input)) {
    return maxWithIndexCpu(
        values, indices, input, axis, keepDims, std::less<>());
  } else {
    throw std::runtime_error(
        "[OneDnnBackend::min] unimplemented for non-CPU engine");
  }
}

Tensor OneDnnBackend::sum(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  return applyReductionOp(
      input, dnnl::algorithm::reduction_sum, axes, keepDims);
}

Tensor OneDnnBackend::cumsum(
    const Tensor& /* input */,
    const unsigned /* axis */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::argmax(
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::argmin(
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::mean(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  return applyReductionOp(
      input, dnnl::algorithm::reduction_mean, axes, keepDims);
}

Tensor OneDnnBackend::median(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::var(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* bias */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::std(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::norm(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    double /* p */ /* = 2 */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::countNonzero(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  std::vector<Dim> dims(input.ndim(), 1);
  auto zero = this->full(Shape(dims), 0, input.type());
  // can't use s32 for sum in OneDNN
  auto res = applyBinop(
      input, zero, dnnl::algorithm::binary_ne, dnnl::memory::data_type::f32);
  return sum(res, axes, keepDims).astype(fl::dtype::s32);
}

Tensor OneDnnBackend::any(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  return countNonzero(input, axes, keepDims) != 0;
}

Tensor OneDnnBackend::all(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::applyReductionOp(
    const Tensor& input,
    const dnnl::algorithm alg,
    const std::vector<int>& axes,
    const bool keepDims) {
  // compute final shape
  std::vector<int> axesToReduce;
  if (axes.empty()) {
    axesToReduce.resize(input.ndim());
    std::iota(axesToReduce.begin(), axesToReduce.end(), 0);
  } else {
    axesToReduce = axes;
  }
  std::vector<Dim> dstDims = input.shape().get();
  for (int axis : axesToReduce) {
    if (axis < 0 || axis >= input.ndim()) {
      std::ostringstream oss;
      oss << "[OneDnnBacked] Invalid axis for reduction: " << axis
          << " for tensor of shape: " << input.shape();
      throw std::invalid_argument(oss.str());
    }
    dstDims[axis] = 1;
  }
  Shape dstShape(dstDims);

  // prepare part of memories
  auto& srcTensor = toOneDnnTensor(input);
  const auto srcMem = srcTensor.memory();
  const auto& srcMemDesc = srcTensor.memoryDesc();
  // OneDNN reduction primitive doesn't allow dim reduction, so we use a memDesc
  // w/o dim reduction for primitive arg, althought the final memory's dims
  // might get reduced
  auto dstArgMemDesc = detail::oneDnnContiguousMemDescFromShape(
      dstShape, srcMemDesc.data_type());

  // prepare reduction primitive
  const auto reductionDesc =
      dnnl::reduction::desc(alg, srcMemDesc, dstArgMemDesc, 0, 0);
  const auto reductionPrimtiveDesc =
      dnnl::reduction::primitive_desc(reductionDesc, engine_);
  const auto reductionPrimitive = dnnl::reduction(reductionPrimtiveDesc);

  // prepare dst memories
  auto dstMemDesc = dstArgMemDesc;
  if (!keepDims) {
    dstShape = Shape(detail::removeIndices(dstShape.get(), axesToReduce));
    dstMemDesc = detail::oneDnnContiguousMemDescFromShape(
        dstShape, srcMemDesc.data_type());
  }
  auto dstMem = dnnl::memory(dstMemDesc, engine_);

  // prepare arguments.
  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, srcMem},
      {DNNL_ARG_DST, dstMem},
  };

  // execute primitive
  reductionPrimitive.execute(stream_->handle(), args);
  return toTensor<OneDnnTensor>(dstShape, std::move(dstMem));
}

void OneDnnBackend::print(const Tensor& tensor) {
  std::cout << "OneDnnTensor" << std::endl
            << tensor.getAdapter<OneDnnTensor>().toString() << std::endl;
}

} // namespace fl
