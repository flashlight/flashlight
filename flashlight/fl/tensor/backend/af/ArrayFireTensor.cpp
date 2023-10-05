/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

#include <cassert>
#include <memory>
#include <stdexcept>
#include <utility>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/af/AdvancedIndex.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

#include <af/arith.h>
#include <af/backend.h>
#include <af/device.h>
#include <af/internal.h>
#include <af/sparse.h>

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

ArrayFireTensor::ArrayFireTensor(af::array&& array, const unsigned numDims)
    : arrayHandle_(std::make_shared<af::array>(std::move(array))),
      numDims_(numDims) {}

ArrayFireTensor::ArrayFireTensor(
    std::shared_ptr<af::array> arr,
    std::vector<af::index>&& afIndices,
    std::vector<detail::IndexType>&& indexTypes,
    const unsigned numDims,
    const bool isFlat)
    : arrayHandle_(arr),
      indices_(std::move(afIndices)),
      indexTypes_(std::move(indexTypes)),
      handle_(IndexedArrayComponent(isFlat)),
      numDims_(numDims) {}

ArrayFireTensor::ArrayFireTensor(
    std::shared_ptr<af::array> arr,
    unsigned numDims)
    : arrayHandle_(arr), numDims_(numDims) {}

ArrayFireTensor::ArrayFireTensor()
    : arrayHandle_(std::make_shared<af::array>()), handle_(ArrayComponent()) {}

ArrayFireTensor::ArrayFireTensor(
    const Shape& shape,
    fl::dtype type,
    const void* ptr,
    Location memoryLocation)
    : arrayHandle_(std::make_shared<af::array>(
          detail::fromFlData(shape, ptr, type, memoryLocation))),
      handle_(ArrayComponent()),
      numDims_(shape.ndim()) {}

ArrayFireTensor::ArrayFireTensor(
    const Dim nRows,
    const Dim nCols,
    const Tensor& values,
    const Tensor& rowIdx,
    const Tensor& colIdx,
    StorageType storageType)
    : arrayHandle_(std::make_shared<af::array>(af::sparse(
          nRows,
          nCols,
          toArray(values),
          toArray(rowIdx),
          toArray(colIdx),
          detail::flToAfStorageType(storageType)))),
      handle_(ArrayComponent()),
      // ArrayFire only supports 2D sparsity
      numDims_(2) {}

unsigned ArrayFireTensor::numDims() const {
  return numDims_;
}

ArrayFireTensor::IndexedArrayComponent::IndexedArrayComponent(
    const bool _isFlat /* = false */)
    : isFlat(_isFlat) {}

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
    auto& idxComp = std::get<IndexedArrayComponent>(handle_);
    arrayHandle_ = std::make_shared<af::array>(detail::condenseIndices(
        idxComp.get(*this),
        /* keepDims = */ false,
        indexTypes_,
        /* isFlat = */ idxComp.isFlat));
    // Clear state
    handle_ = ArrayComponent(); // set to passthrough
    indices_ = {}; // remove indices
    indexTypes_ = {}; // remove IndexTypes
  }
  return *arrayHandle_;
}

std::unique_ptr<TensorAdapterBase> ArrayFireTensor::clone() const {
  af::array arr = getHandle(); // increment internal AF refcount
  return std::make_unique<ArrayFireTensor>(
      std::move(arr), numDims());
}

Tensor ArrayFireTensor::copy() {
  getHandle(); // if this tensor was a view, run indexing and promote
  return toTensor<ArrayFireTensor>(arrayHandle_->copy(), numDims());
}

Tensor ArrayFireTensor::shallowCopy() {
  getHandle(); // if this tensor was a view, run indexing and promote
  return Tensor(std::unique_ptr<ArrayFireTensor>(
      new ArrayFireTensor(arrayHandle_, numDims())));
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
  detail::afToFlDims(getHandle().dims(), numDims(), shape_);
  return shape_;
}

fl::dtype ArrayFireTensor::type() {
  return detail::afToFlType(getHandle().type());
}

bool ArrayFireTensor::isSparse() {
  return getHandle().issparse();
}

af::dtype ArrayFireTensor::afHandleType() {
  return arrayHandle_->type();
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

void ArrayFireTensor::host(void* out) {
  AF_CHECK(af_get_data_ptr(out, getHandle().get()));
}

void ArrayFireTensor::unlock() {
  AF_CHECK(af_unlock_array(getHandle().get()));
}

bool ArrayFireTensor::isLocked() {
  bool res;
  auto err = af_is_locked_array(&res, getHandle().get());
  if (err != AF_SUCCESS) {
    throw std::runtime_error(
        "ArrayFireTensor::isLocked - af_is_locked_array returned error: " +
        std::to_string(err));
  }
  return res;
}

bool ArrayFireTensor::isContiguous() {
  return af::isLinear(getHandle());
}

Shape ArrayFireTensor::strides() {
  return detail::afToFlDims(af::getStrides(getHandle()), numDims());
}

const Stream& ArrayFireTensor::stream() const {
  // TODO indexing is unlikely to change the stream associated with a tensor.
  // But if it can, we need to call `getHandle()` here.
  return ArrayFireBackend::getInstance().getStreamOfArray(*arrayHandle_);
}

Tensor ArrayFireTensor::astype(const dtype type) {
  auto a = getHandle().as(detail::flToAfType(type));
  return toTensor<ArrayFireTensor>(std::move(a), numDims());
}

Tensor ArrayFireTensor::index(const std::vector<Index>& indices) {
  if (indices.size() > AF_MAX_DIMS) {
    throw std::invalid_argument(
        "ArrayFire-backed tensor was indexed with > 4 elements:"
        "ArrayFire tensors support up to 4 dimensions.");
  }

  // TODO: vet and stress test this a lot more/add proper support for
  // multi-tensor
  // If indexing by a single element and it's a tensor with the same number of
  // indices as the array being indexed, do a flat index as this is probably a
  // filter-based index (for example: a(a < 5)).
  bool completeTensorIndex = indices.size() == 1 &&
      indices.front().type() == detail::IndexType::Tensor &&
      indices.front().get<Tensor>().elements() == getHandle().elements();
  std::vector<af::index> afIndices;
  if (completeTensorIndex) {
    afIndices = {af::index(0)};
  } else {
    afIndices = {af::span, af::span, af::span, af::span}; // implicit spans
  }

  if (indices.size() > afIndices.size()) {
    throw std::logic_error(
        "ArrayFireTensor::index internal error - passed indiecs is larger "
        "than the number of af indices");
  }

  // Fill in corresponding index types for each af index
  std::vector<detail::IndexType> indexTypes(afIndices.size());
  size_t i = 0;
  for (; i < indices.size(); ++i) {
    indexTypes[i] = indices[i].type();
    afIndices[i] = detail::flToAfIndex(indices[i]);
  }
  // If we're adding implicit spans, fill those indexTypes in
  for (; i < afIndices.size(); ++i) {
    indexTypes[i] = detail::IndexType::Span;
  }

  getHandle(); // if this tensor was a view, run indexing and promote

  assert(afIndices.size() == indexTypes.size());
  // Compute numDums for the new Tensor
  unsigned newNumDims = numDims();

  if (completeTensorIndex) {
    // TODO/FIXME: compute this based on the number of els in the indexing
    // tensor(s)
    newNumDims = 1;
  } else {
    for (const auto& type : indexTypes) {
      if (type == detail::IndexType::Literal) {
        newNumDims--;
      }
    }
  }
  newNumDims = std::max(newNumDims, 1u); // can never index to a 0 dim tensor

  return fl::Tensor(std::unique_ptr<ArrayFireTensor>(new ArrayFireTensor(
      arrayHandle_,
      std::move(afIndices),
      std::move(indexTypes),
      newNumDims,
      /* isFlat = */ false)));
}

Tensor ArrayFireTensor::flatten() const {
  return toTensor<ArrayFireTensor>(af::flat(getHandle()), /* numDims = */ 1);
}

Tensor ArrayFireTensor::flat(const Index& idx) const {
  getHandle(); // if this tensor was a view, run indexing and promote
  // Return a lazy indexing operation. Indexing with a single index on an
  // ArrayFire tensor (with a type that is not an af::array) ends up doing
  // flat indexing, so all index assignment operators will work as they are.
  return fl::Tensor(std::unique_ptr<ArrayFireTensor>(new ArrayFireTensor(
      arrayHandle_,
      {detail::flToAfIndex(idx)},
      {idx.type()},
      /* numDims = */ 1,
      /* isFlat = */ true)));
}

Tensor ArrayFireTensor::asContiguousTensor() {
  if (isContiguous()) {
    af::array other = getHandle();
    return toTensor<ArrayFireTensor>(std::move(other), numDims());
  }

  const af::array& array = getHandle();
  auto linearArray = af::array(array.dims(), array.type());
  af::copy(linearArray, array, af::span);
  return toTensor<ArrayFireTensor>(std::move(linearArray), numDims());
}

void ArrayFireTensor::setContext(void* context) {} // noop

void* ArrayFireTensor::getContext() {
  return nullptr; // noop
}

std::string ArrayFireTensor::toString() {
  const char* afStr = af::toString("ArrayFireTensor", getHandle());
  // std::string copies `afStr` content into its own buffer
  const std::string str(afStr);
  af::freeHost(afStr);
  return str;
}

std::ostream& ArrayFireTensor::operator<<(std::ostream& ostr) {
  ostr << this->toString();
  return ostr;
}

/******************** Assignment Operators ********************/
#define ASSIGN_OP_TYPE(FUN, AF_OP, TYPE)                                 \
  void ArrayFireTensor::FUN(const TYPE& val) {                           \
    std::visit(                                                          \
        [val, this](auto&& arr) { arr.get(*this) AF_OP val; }, handle_); \
  }

#define ASSIGN_OP_LITERALS(FUN, AF_OP)        \
  ASSIGN_OP_TYPE(FUN, AF_OP, double);         \
  ASSIGN_OP_TYPE(FUN, AF_OP, float);          \
  ASSIGN_OP_TYPE(FUN, AF_OP, int);            \
  ASSIGN_OP_TYPE(FUN, AF_OP, unsigned);       \
  ASSIGN_OP_TYPE(FUN, AF_OP, bool);           \
  ASSIGN_OP_TYPE(FUN, AF_OP, char);           \
  ASSIGN_OP_TYPE(FUN, AF_OP, unsigned char);  \
  ASSIGN_OP_TYPE(FUN, AF_OP, short);          \
  ASSIGN_OP_TYPE(FUN, AF_OP, unsigned short); \
  ASSIGN_OP_TYPE(FUN, AF_OP, long);           \
  ASSIGN_OP_TYPE(FUN, AF_OP, unsigned long);  \
  ASSIGN_OP_TYPE(FUN, AF_OP, long long);      \
  ASSIGN_OP_TYPE(FUN, AF_OP, unsigned long long);

af::array ArrayFireTensor::adjustInPlaceOperandDims(const Tensor& operand) {
  // optimstically try to moddims the operand's singleton dims
  const af::dim4& preIdxDims = arrayHandle_->dims();
  const af::array& operandArr = toArray(operand);

  // dims to which to try to modify the input if doing indexing
  af::dim4 newDims;
  const af::dim4 operandDims = operandArr.dims();

  using detail::IndexType;
  if (indices_ && indices_.value().size() == 1) {
    // This case is only reachable via tensor-based indexing or indexing on a
    // tensor via Tensor::flat()
    if (numDims_ != 1) {
      throw std::invalid_argument(
          "ArrayFireTensor::adjustInPlaceOperandDims "
          "index size was 1 but tensor has greater than 1 dimension.");
    }
  } else if (indices_ && !indices_.value().empty()) {
    // All other indexing operations
    const auto& indices = indices_.value();
    const auto& indexTypes = indexTypes_.value();
    if (indices.size() != indexTypes.size()) {
      throw std::invalid_argument(
          "ArrayFireTensor adjustInPlaceOperandDims - passed indices"
          " and indexTypes are of different sizes.");
    }

    // If the dimensions being indexed are 1 and collapsing them yields the same
    // shape as the operand, we can safely moddims, the operand, else there's a
    // dimension mismatch. For example:
    // {4, 5, 6, 7}(span, span, 5) --> {4, 5, 1, 7} --> {4, 5, 7}
    // {4, 5, 6, 7}(4) --> {1, 5, 1, 7} --> {5, 1, 7, 1}
    std::vector<unsigned> indicesToCompress;
    for (unsigned i = 0; i < indices.size(); ++i) {
      // If an index literal, the corresponding dimension in the indexed array
      // is 1, then we indexed the input to a dim of 1, so we can condense that
      // index
      if (indexTypes[i] == IndexType::Literal) {
        indicesToCompress.push_back(i);
      }
    }

    af::dim4 condensedDims(1, 1, 1, 1);
    af::dim4 postIdxDims = preIdxDims;
    unsigned outDimIdx = 0;
    unsigned compressIdx = 0;
    for (unsigned i = 0; i < AF_MAX_DIMS; ++i) {
      if (compressIdx < indicesToCompress.size() &&
          i == indicesToCompress[compressIdx]) {
        compressIdx++;
        postIdxDims[i] = 1;
      } else {
        // Use the size of the dim post-indexing. Span uses the preIdx dim
        // and literals are pushed to 1.
        if (i < indexTypes.size()) {
          if (indexTypes[i] == IndexType::Tensor) {
            dim_t size;
            AF_CHECK(af_get_elements(&size, indices[i].get().idx.arr));
            postIdxDims[i] = size;
          } else if (indexTypes[i] == IndexType::Range) {
            postIdxDims[i] = af::seq(indices[i].get().idx.seq).size;
          } else if (indexTypes[i] == IndexType::Literal) {
            postIdxDims[i] = 1;
          }
        }
        condensedDims[outDimIdx] = postIdxDims[i];
        outDimIdx++;
      }
    }

    // Can modify the operand to work with the proxy or array input only by
    // removing singleton dimensions
    if (condensedDims == operandDims) {
      newDims = postIdxDims;
    } else {
      throw std::invalid_argument(
          "ArrayFireTensor adjustInPlaceOperandDims: can't apply operation "
          "in-place to indexed ArrayFireTensor - dimensions don't match.");
    }
  } else {
    // No indexing so no change in dimensions required
    newDims = operandDims;
  }

  // af::moddims involves an eval. This will be fixed in AF 3.8.1/3.8.2
  bool doModdims = operandArr.dims() != newDims;
  return (doModdims ? af::moddims(operandArr, newDims) : operandArr);
}

#define ASSIGN_OP_TENSOR(FUN, AF_OP)                                   \
  void ArrayFireTensor::FUN(const Tensor& tensor) {                    \
    std::visit(                                                        \
        [&tensor, this](auto&& arr) {                                  \
          arr.get(*this) AF_OP this->adjustInPlaceOperandDims(tensor); \
        },                                                             \
        handle_);                                                      \
  }

#define ASSIGN_OP(FUN, AF_OP)  \
  ASSIGN_OP_TENSOR(FUN, AF_OP) \
  ASSIGN_OP_LITERALS(FUN, AF_OP)

// (function name, AF op). Use build-in AF operators.
ASSIGN_OP(inPlaceSubtract, -=);
ASSIGN_OP(inPlaceMultiply, *=);
ASSIGN_OP(inPlaceDivide, /=);

// Instantiate definitions for type literals - those remain unchanged:
ASSIGN_OP_LITERALS(assign, =);
void ArrayFireTensor::assign(const Tensor& tensor) {
  std::visit(
      [&tensor, this](auto&& arr) {
        if (indices_) {
          // If this is an indexing op, do as other in-place ops with lvalue
          // temporaries as a result of indexing do
          arr.get(*this) = this->adjustInPlaceOperandDims(tensor);
        } else {
          // Not an indexing op - just assign the tensor, but make sure to
          // update the number of dims
          arr.get(*this) = toArray(tensor);
          this->numDims_ = tensor.ndim();
        }
      },
      handle_);
}

/*
 * A custom advanced index kernel is used for the in-place-add operator.
 *
 * If performing someTensor(index1, ...) += anotherTensor, call this kernel, as
 * it properly-handles the case of repeated indices.
 */
// Instantiate definitions for type literals - those remain unchanged:
ASSIGN_OP_LITERALS(inPlaceAdd, +=);
// Special tensor op:
void ArrayFireTensor::inPlaceAdd(const Tensor& tensor) {
  // First, check if this a tensor that's going to be lazily indexed. Don't
  // implicitly cast to an array, else that will trigger indexing.
  // Carefully get the handle types without calling type(), which will lazily
  // evaluate indexing
  af::dtype operandHandleType =
      tensor.getAdapter<ArrayFireTensor>().afHandleType();
  af::dtype handleType = arrayHandle_->type();
  // not all types are compatible with the kernel
  bool typeIncompatible =
      (handleType != af::dtype::f32 && handleType != af::dtype::f16) ||
      (operandHandleType != af::dtype::f32 &&
       operandHandleType != af::dtype::f16);
  if (!std::holds_alternative<IndexedArrayComponent>(handle_) ||
      typeIncompatible ||
      !FL_BACKEND_CUDA // TODO{fl::Tensor} advanced indexing only impl for CUDA
  ) {
    // Call the regular af::array::operator+=
    std::visit(
        [&tensor, this](auto&& arr) {
          arr.get(*this) += this->adjustInPlaceOperandDims(tensor);
        },
        handle_);
    return;
  } else {
    af::dim4 inDims = arrayHandle_->dims();
    af::dim4 idxStart;
    af::dim4 idxEnd;
    std::vector<af::array> idxArr(4);
    auto idxFunc = [&idxStart, &idxEnd, &idxArr, &inDims](
                       const af::index& index, int pos) {
      if (index.isspan()) {
        idxStart[pos] = 0;
        idxEnd[pos] = inDims[pos];
      } else {
        const auto& idxSeq = index.get();
        if (idxSeq.isSeq) {
          // arrayfire uses inclusive last dimension, we use exclusive
          idxStart[pos] = idxSeq.idx.seq.begin;
          idxEnd[pos] = idxSeq.idx.seq.end + 1;
        } else {
          af_array arr;
          af_retain_array(&arr, idxSeq.idx.arr);
          idxArr[pos] = af::array(arr);
          idxStart[pos] = 0;
          idxEnd[pos] = idxArr[pos].dims(0);
        }
      }
    };

    unsigned i = 0;
    for (; i < indices_.value().size(); ++i) {
      idxFunc(indices_.value()[i], i);
    }
    // The kernel needs to be padded with spans for remaining dims
    for (; i < AF_MAX_DIMS; ++i) {
      idxFunc(af::span, i);
    }

    fl::detail::advancedIndex(
        toArray(tensor), idxStart, idxEnd, inDims, idxArr, *arrayHandle_);
  }
}

#undef ASSIGN_OP_TYPE
#undef ASSIGN_OP_LITERALS
#undef ASSIGN_OP_TENSOR
#undef ASSIGN_OP

} // namespace fl
