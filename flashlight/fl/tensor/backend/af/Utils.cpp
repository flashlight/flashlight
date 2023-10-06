/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/Utils.h"

#include <stdexcept>
#include <unordered_map>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

namespace fl::detail {

af::dtype flToAfType(fl::dtype type) {
  static const std::unordered_map<fl::dtype, af::dtype>
      kFlashlightTypeToArrayFire = {
          {fl::dtype::f16, af::dtype::f16},
          {fl::dtype::f32, af::dtype::f32},
          {fl::dtype::f64, af::dtype::f64},
          {fl::dtype::b8, af::dtype::b8},
          {fl::dtype::s16, af::dtype::s16},
          {fl::dtype::s32, af::dtype::s32},
          {fl::dtype::s64, af::dtype::s64},
          {fl::dtype::u8, af::dtype::u8},
          {fl::dtype::u16, af::dtype::u16},
          {fl::dtype::u32, af::dtype::u32},
          {fl::dtype::u64, af::dtype::u64}};
  return kFlashlightTypeToArrayFire.at(type);
}

fl::dtype afToFlType(af::dtype type) {
  static const std::unordered_map<af::dtype, fl::dtype>
      kArrayFireTypeToFlashlight = {
          {af::dtype::f16, fl::dtype::f16},
          {af::dtype::f32, fl::dtype::f32},
          {af::dtype::f64, fl::dtype::f64},
          {af::dtype::b8, fl::dtype::b8},
          {af::dtype::s16, fl::dtype::s16},
          {af::dtype::s32, fl::dtype::s32},
          {af::dtype::s64, fl::dtype::s64},
          {af::dtype::u8, fl::dtype::u8},
          {af::dtype::u16, fl::dtype::u16},
          {af::dtype::u32, fl::dtype::u32},
          {af::dtype::u64, fl::dtype::u64}};
  return kArrayFireTypeToFlashlight.at(type);
}

af_mat_prop flToAfMatrixProperty(MatrixProperty property) {
  switch (property) {
    case MatrixProperty::None:
      return AF_MAT_NONE;
    case MatrixProperty::Transpose:
      return AF_MAT_TRANS;
    default:
      throw std::invalid_argument(
          "flToAfMatrixProperty: invalid property specified");
  }
}

af_storage flToAfStorageType(StorageType storageType) {
  switch (storageType) {
    case StorageType::Dense:
      return AF_STORAGE_DENSE;
    case StorageType::CSR:
      return AF_STORAGE_CSR;
    case StorageType::CSC:
      return AF_STORAGE_CSC;
    case StorageType::COO:
      return AF_STORAGE_COO;
    default:
      throw std::invalid_argument(
          "flToAfStorageType: Flashlight storage type "
          "doesn't have an ArrayFire analog");
  }
}

af_topk_function flToAfTopKSortMode(SortMode sortMode) {
  switch (sortMode) {
    case SortMode::Descending:
      return AF_TOPK_MAX;
    case SortMode::Ascending:
      return AF_TOPK_MIN;
    default:
      throw std::invalid_argument(
          "flToAfTopKSortMode: sort mode with no ArrayFire analog specified");
  }
}

af::dim4 flToAfDims(const Shape& shape) {
  if (shape.ndim() > 4) {
    throw std::invalid_argument(
        "flToAfDims: ArrayFire shapes can't be more than 4 dimensions");
  }

  af::dim4 out(1, 1, 1, 1);
  for (size_t i = 0; i < shape.ndim(); ++i) {
    out.dims[i] = shape.dim(i);
  }
  return out;
}

void afToFlDims(const af::dim4& d, const unsigned numDims, Shape& s) {
  if (numDims > AF_MAX_DIMS) {
    throw std::invalid_argument("afToFlDims - numDims > AF_MAX_DIMS");
  }

  auto& storage = s.get();

  // numdims constraint is enforced by the internal API per condenseDims
  if (numDims == 1 && d.elements() == 0) {
    // Empty tensor
    storage.resize(1);
    s[0] = 0;
    return;
  }

  // numDims == 0 --> scalar tensor
  if (numDims == 0) {
    storage.resize(0);
    return;
  }

  storage.resize(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    s[i] = d[i];
  }
}

Shape afToFlDims(const af::dim4& d, const unsigned numDims) {
  Shape s;
  afToFlDims(d, numDims, s);
  return s;
}

af::seq flRangeToAfSeq(const fl::range& range) {
  const int start = range.start();
  const auto& optEnd = range.end();
  const int end = optEnd.has_value() ? optEnd.value() - 1 : af::end;
  // There could be have other empty sequence representations, e.g., (0, -1)
  // for axis with 1 element. In those cases, AF will throw internally --
  // we can't throw here because these cases  axis-size dependent.
  if (optEnd.has_value() && optEnd.value() == start) {
    throw std::runtime_error(
        "flRangeToAfSeq: AF seq can't represent empty sequence");
  }
  return af::seq(start, end, range.stride());
}

af::index flToAfIndex(const fl::Index& idx) {
  switch (idx.type()) {
    case IndexType::Tensor:
      return af::index(toArray(idx.get<Tensor>()));
    case IndexType::Span:
      return af::index(af::span);
    case IndexType::Range:
      return af::index(flRangeToAfSeq(idx.get<range>()));
    case IndexType::Literal:
      return af::index(idx.get<Dim>());
    default:
      throw std::invalid_argument(
          "flToAfIndex: fl::Index has unknown or invalid type.");
  }
}

af::dim4 condenseDims(const af::dim4& dims) {
  if (dims.elements() == 0) {
    return af::dim4(0);
  }

  // Find the condensed shape
  af::dim4 newDims(1, 1, 1, 1);
  unsigned newDimIdx = 0;
  for (unsigned i = 0; i < AF_MAX_DIMS; ++i) {
    if (dims[i] != 1) {
      // found a non-1 dim size - populate newDims
      newDims[newDimIdx] = dims[i];
      newDimIdx++;
    }
  }
  return newDims;
}

af::array condenseIndices(
    const af::array& arr,
    const bool keepDims /* = false */,
    const std::optional<std::vector<detail::IndexType>>& indexTypes /* = {} */,
    const bool isFlat /* = false */) {
  // Fast path - return the Array as is if keepDims - don't consolidate
  if (keepDims) {
    return arr;
  }
  // Fast path - Array has zero elements or a dim of size zero
  if (arr.elements() == 0) {
    return arr;
  }

  const af::dim4& dims = arr.dims();
  af::dim4 newDims(1, 1, 1, 1);
  unsigned newDimIdx = 0;
  for (unsigned i = 0; i < AF_MAX_DIMS; ++i) {
    // If we're doing an index op (indexTypes is non-empty), then only collapse
    // the dimension if it contains an index literal and we aren't doing flat
    // indexing (which collapses all dims)
    if (dims[i] == 1 && indexTypes && indexTypes.value().size() > i &&
        indexTypes.value()[i] != detail::IndexType::Literal && !isFlat) {
      newDims[newDimIdx] = 1;
      newDimIdx++;
    } else if (dims[i] != 1) {
      // found a non-1 dim size - populate newDims.
      newDims[newDimIdx] = dims[i];
      newDimIdx++;
    }
  }

  // Only change dims if condensing is possible
  if (newDims != arr.dims()) {
    return af::moddims(arr, newDims);
  } else {
    return arr;
  }
}

af_source flToAfLocation(Location location) {
  switch (location) {
    case Location::Host:
      return afHost;
    case Location::Device:
      return afDevice;
    default:
      throw std::invalid_argument(
          "flToAfLocation: no valid ArrayFire location exists "
          " for given Flashlight location.");
  }
}

af::array fromFlData(
    const Shape& shape,
    const void* ptr,
    fl::dtype type,
    fl::Location memoryLocation) {
  af::dim4 dims = detail::flToAfDims(shape);
  af::dtype afType = detail::flToAfType(type);
  af_source loc = detail::flToAfLocation(memoryLocation);

  // No or null buffer
  if (!ptr) {
    return af::array(dims, afType);
  }

  using af::dtype;
  switch (afType) {
    case f32:
      return af::array(dims, reinterpret_cast<const float*>(ptr), loc);
    case f64:
      return af::array(dims, reinterpret_cast<const double*>(ptr), loc);
    case s32:
      return af::array(dims, reinterpret_cast<const int*>(ptr), loc);
    case u32:
      return af::array(dims, reinterpret_cast<const unsigned*>(ptr), loc);
    case s64:
      return af::array(dims, reinterpret_cast<const long long*>(ptr), loc);
    case u64:
      return af::array(
          dims, reinterpret_cast<const unsigned long long*>(ptr), loc);
    case s16:
      return af::array(dims, reinterpret_cast<const short*>(ptr), loc);
    case u16:
      return af::array(dims, reinterpret_cast<const unsigned short*>(ptr), loc);
    case b8:
      return af::array(dims, reinterpret_cast<const char*>(ptr), loc);
    case u8:
      return af::array(dims, reinterpret_cast<const unsigned char*>(ptr), loc);
    default:
      throw std::invalid_argument(
          "fromFlData: can't construct ArrayFire array from given type.");
  }
}

af_border_type flToAfPadType(PadType type) {
  switch (type) {
    case PadType::Constant:
      return AF_PAD_ZERO; // constant padding --> zero padding in AF
    case PadType::Edge:
      return AF_PAD_CLAMP_TO_EDGE;
    case PadType::Symmetric:
      return AF_PAD_SYM;
    default:
      throw std::invalid_argument(
          "flToAfPadType: Flashlight padding "
          "type not supported by ArrayFire");
  }
}

} // namespace fl
