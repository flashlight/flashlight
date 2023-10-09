
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/onednn/Utils.h"

#include <numeric>
#include <stdexcept>

#include <dnnl_debug.h>

namespace fl::detail {

namespace {

template <typename K, typename V>
std::unordered_map<V, K> invertMap(const std::unordered_map<K, V>& map) {
  std::unordered_map<V, K> invertedMap;
  for (const auto& [key, val] : map) {
    invertedMap.emplace(val, key);
  }
  return invertedMap;
}

// wrapped inside a function to avoid strange static initialization issue
const std::unordered_map<fl::dtype, dnnl::memory::data_type>&
getFlashlightTypeToOnednnTypeMap() {
  static const std::unordered_map<fl::dtype, dnnl::memory::data_type>
      kFlashlightTypeToOneDnnType = {
          {fl::dtype::f16, dnnl::memory::data_type::f16},
          {fl::dtype::f32, dnnl::memory::data_type::f32},
          {fl::dtype::f64, dnnl::memory::data_type::f64},
          {fl::dtype::b8, dnnl::memory::data_type::s8},
          {fl::dtype::u8, dnnl::memory::data_type::u8},
          {fl::dtype::s32, dnnl::memory::data_type::s32},
      };
  return kFlashlightTypeToOneDnnType;
}

} // namespace

dnnl::memory::data_type flToOneDnnType(const fl::dtype type) {
  const auto& flToOneDnnTypeMap = getFlashlightTypeToOnednnTypeMap();
  const auto& iter = flToOneDnnTypeMap.find(type);
  if (iter == flToOneDnnTypeMap.end()) {
    throw std::invalid_argument(
        "FL type unsupported in OneDNN backend: " + dtypeToString(type));
  }
  return iter->second;
}

fl::dtype oneDnnToFlType(const dnnl::memory::data_type type) {
  static const std::unordered_map<dnnl::memory::data_type, fl::dtype>
      kOneDnnTypeToFlashlighType =
          invertMap(getFlashlightTypeToOnednnTypeMap());

  const auto& iter = kOneDnnTypeToFlashlighType.find(type);
  if (iter == kOneDnnTypeToFlashlighType.end()) {
    throw std::invalid_argument(
        "FL type unsupported in OneDNN backend: " + oneDnnDataTypeToStr(type));
  }
  return iter->second;
}

bool isTypeSupportedByOneDnn(const fl::dtype type) {
  return getFlashlightTypeToOnednnTypeMap().count(type) != 0;
}

std::string oneDnnDataTypeToStr(const dnnl::memory::data_type type) {
  return dnnl_dt2str(dnnl::memory::convert_to_c(type));
}

dnnl::memory::dims shapeToOneDnnStrides(const Shape& shape) {
  // NOTE this conforms to the existing limit imposed by the ArrayFire backend.
  // We could easily relax this -- OneDNN supports up to 12 dimensions.
  if (shape.ndim() > 4) {
    throw std::invalid_argument(
        "OneDNN expects maximum dimension of 4, but got: " +
        std::to_string(shape.ndim()));
  }
  // Recall that dims are reversed, see docs in OneDnnTensor.
  dnnl::memory::dims strides{1};
  for (int idx = 0; idx < shape.ndim() - 1; idx++) {
    strides.push_back(strides.back() * shape.dim(idx));
  }
  std::reverse(strides.begin(), strides.end());
  return strides;
}

dnnl::memory::dims flDimsToOneDnnDims(const std::vector<Dim>& flDims) {
  if (flDims.empty()) {
    return {1}; // scalar, OneDNN memory returns null handle with {} as dims
  }
  return dnnl::memory::dims(flDims.rbegin(), flDims.rend());
}

dnnl::memory::dims shapeToOneDnnDims(const Shape& shape) {
  return flDimsToOneDnnDims(shape.get());
}

Shape oneDnnDimsToShape(const dnnl::memory::dims& dims) {
  return Shape(std::vector<Dim>(dims.rbegin(), dims.rend()));
}

bool isFpType(dnnl::memory::data_type type) {
  switch (type) {
    case dnnl::memory::data_type::f32:
    case dnnl::memory::data_type::f16:
    case dnnl::memory::data_type::bf16:
      return true;
    default:
      return false;
  }
}

bool isIntType(dnnl::memory::data_type type) {
  return !isFpType(type);
}

dnnl::memory::data_type getTypeWithLargerRange(
    dnnl::memory::data_type t1,
    dnnl::memory::data_type t2) {
  if ((isFpType(t1) && isFpType(t2)) || (isIntType(t1) && isIntType(t2))) {
    auto t1Size = dnnl::memory::data_type_size(t1);
    auto t2Size = dnnl::memory::data_type_size(t2);
    return t1Size >= t2Size ? t1 : t2;
  }
  return isFpType(t1) ? t1 : t2;
}

dnnl::memory::desc oneDnnContiguousMemDescFromShape(
    const Shape& shape,
    const dnnl::memory::data_type type) {
  return dnnl::memory::desc(
      detail::shapeToOneDnnDims(shape),
      type,
      detail::shapeToOneDnnStrides(shape),
      /* allowEmpty */ true);
}

} // namespace fl
