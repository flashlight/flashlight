/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/onednn/OneDnnBackend.h"

#include <cassert>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"
#include "flashlight/fl/tensor/backend/onednn/Utils.h"

#define FL_ONEDNN_BACKEND_UNIMPLEMENTED \
  throw std::invalid_argument(          \
      "OneDnnBackend::" + std::string(__func__) + " - unimplemented.");

namespace fl {

namespace {

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

dnnl::memory::desc getMemDescWithLargerDataRange(
  const dnnl::memory::desc& memDesc1,
  const dnnl::memory::desc& memDesc2) {
  const auto dataType = detail::getTypeWithLargerRange(
      memDesc1.data_type(), memDesc2.data_type());
  return dataType == memDesc1.data_type() ? memDesc1 : memDesc2;
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
    std::optional<dnnl::memory::data_type> dstType) {
  // some common fast paths
  if (lhsShape == rhsShape) {
    return {
        .dstMemDesc = dstType.has_value()
            ? detail::copyMemDescWithNewType(lhsMemDesc, dstType.value())
            : getMemDescWithLargerDataRange(lhsMemDesc, rhsMemDesc),
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
  // allow implicit casting
  const auto dataType = dstType.value_or(detail::getTypeWithLargerRange(
      lhsMemDesc.data_type(), rhsMemDesc.data_type()));
  Shape dstShape(dstDims);
  return {
      .dstMemDesc = dnnl::memory::desc(
          detail::shapeToOneDnnDims(dstShape),
          dataType,
          detail::shapeToOneDnnStrides(dstShape)),
      .dstShape = dstShape};
}

} // namespace

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

const dnnl::stream& OneDnnBackend::nativeStream() const {
  return stream_->handle();
}

const dnnl::engine& OneDnnBackend::engine() const {
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

void OneDnnBackend::setSeed(const int /* seed */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::randn(const Shape& /* shape */, dtype /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::rand(const Shape& /* shape */, dtype /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
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

Tensor OneDnnBackend::identity(const Dim /* dim */, const dtype /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::arange(
    const Shape& /* shape */,
    const Dim /* seqDim */,
    const dtype /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::iota(
    const Shape& /* dims */,
    const Shape& /* tileDims */,
    const dtype /* type */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

/************************ Shaping and Indexing *************************/
Tensor OneDnnBackend::reshape(
    const Tensor& /* tensor */,
    const Shape& /* shape */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
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
  auto& srcMem = tensor.getAdapter<OneDnnTensor>().memory();
  const auto& srcMemDesc = srcMem.get_desc();
  const auto srcMemDims = srcMemDesc.dims();
  const auto dstMemDesc =
      srcMemDesc.reshape(detail::shapeToOneDnnDims(newShape));
  auto dstMem = dnnl::memory(dstMemDesc, engine_);

  // prepare primitive
  const auto reorderDstStrides =
      getStridesAfterPermuteAxes(srcMemDims, oldToNewAxes);
  const auto reorderDstMemDesc =
      dnnl::memory::desc(srcMemDims, srcMemDesc.data_type(), reorderDstStrides);
  const auto reorderPrimitiveDesc = dnnl::reorder::primitive_desc(
      engine_, srcMemDesc, engine_, reorderDstMemDesc);
  const auto reorderPrimitive = dnnl::reorder(reorderPrimitiveDesc);

  // execute primitive
  reorderPrimitive.execute(stream_->handle(), srcMem, dstMem);
  return toTensor<OneDnnTensor>(newShape, std::move(dstMem));
}

Tensor OneDnnBackend::tile(
    const Tensor& /* tensor */,
    const Shape& /* shape */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
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

Tensor OneDnnBackend::exp(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::log(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::negative(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::logicalNot(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
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

Tensor OneDnnBackend::sqrt(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::tanh(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::floor(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::ceil(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::rint(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::absolute(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::sigmoid(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::erf(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
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

Tensor OneDnnBackend::roll(
    const Tensor& /* tensor */,
    const int /* shift */,
    const unsigned /* axis */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::isnan(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::isinf(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::sign(const Tensor& /* tensor */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
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

FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(||, logicalOr);
FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(&&, logicalAnd);
FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(%, mod);
FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(&, bitwiseAnd);
FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(|, bitwiseOr);
FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(^, bitwiseXor);
FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(<<, lShift);
FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF(>>, rShift);
#undef FL_ONEDNN_BINARY_OP_UNSUPPORTED_DEF
#undef FL_ONEDNN_BINARY_OP_TYPE_UNSUPPORTED_DEF
#undef FL_ONEDNN_BINARY_OP_LITERALS_UNSUPPORTED_DEF

#define FL_ONEDNN_BINARY_OP_LITERALS_DEF(FUNC, TYPE, CAST_TYPE)         \
  Tensor OneDnnBackend::FUNC(const Tensor& a, TYPE rhs) {               \
    CAST_TYPE val = static_cast<CAST_TYPE>(rhs);                        \
    Shape literalShape(std::vector<Dim>(a.ndim(), 1));                  \
    auto type = dtype_traits<CAST_TYPE>::fl_type;                       \
    auto rhsTensor =                                                    \
        toTensor<OneDnnTensor>(literalShape, type, &val, a.location()); \
    return FUNC(a, rhsTensor);                                          \
  }                                                                     \
  Tensor OneDnnBackend::FUNC(TYPE lhs, const Tensor& a) {               \
    CAST_TYPE val = static_cast<CAST_TYPE>(lhs);                        \
    Shape literalShape(std::vector<Dim>(a.ndim(), 1));                  \
    auto type = dtype_traits<CAST_TYPE>::fl_type;                       \
    auto lhsTensor =                                                    \
        toTensor<OneDnnTensor>(literalShape, type, &val, a.location()); \
    return FUNC(lhsTensor, a);                                          \
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

Tensor OneDnnBackend::minimum(const Tensor& lhs, const Tensor& rhs) {
  return applyBinop(lhs, rhs, dnnl::algorithm::binary_min);
}

Tensor OneDnnBackend::maximum(const Tensor& lhs, const Tensor& rhs) {
  return applyBinop(lhs, rhs, dnnl::algorithm::binary_max);
}
Tensor OneDnnBackend::applyBinop(
    const Tensor& lhs,
    const Tensor& rhs,
    dnnl::algorithm alg,
    std::optional<dnnl::memory::data_type> dstType /* = std::nullopt */) {
  // prepare memories
  auto& lhsMem = toOneDnnTensor(lhs).memory();
  auto& rhsMem = toOneDnnTensor(rhs).memory();
  const auto lhsMemDesc = lhsMem.get_desc();
  const auto rhsMemDesc = rhsMem.get_desc();
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

/************************** BLAS ***************************/

Tensor OneDnnBackend::matmul(
    const Tensor& /* lhs */,
    const Tensor& /* rhs */,
    MatrixProperty /* lhsProp */,
    MatrixProperty /* rhsProp */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

/************************** Reductions ***************************/

Tensor OneDnnBackend::amin(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::amax(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

void OneDnnBackend::min(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

void OneDnnBackend::max(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::sum(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
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
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
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
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::any(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

Tensor OneDnnBackend::all(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_ONEDNN_BACKEND_UNIMPLEMENTED;
}

void OneDnnBackend::print(const Tensor& tensor) {
  std::cout << "OneDnnTensor" << std::endl
            << tensor.getAdapter<OneDnnTensor>().toString() << std::endl;
}

} // namespace fl
