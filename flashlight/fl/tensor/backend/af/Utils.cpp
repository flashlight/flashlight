/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/Utils.h"

#include <stdexcept>
#include <unordered_map>

namespace fl {
namespace detail {

const std::unordered_map<fl::dtype, af::dtype> kFlashlightTypeToArrayFire = {
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

const std::unordered_map<af::dtype, fl::dtype> kArrayFireTypeToFlashlight = {
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

af::dtype flToAfType(fl::dtype type) {
  return kFlashlightTypeToArrayFire.at(type);
}

fl::dtype afToFlType(af::dtype type) {
  return kArrayFireTypeToFlashlight.at(type);
}

af::dim4 flToAfDims(const Shape& shape) {
  if (shape.nDims() > 4) {
    throw std::invalid_argument(
        "flToAfDims: ArrayFire shapes can't be more than 4 dimensions");
  }
  af::dim4 out(1, 1, 1, 1);
  for (size_t i = 0; i < shape.nDims(); ++i) {
    out.dims[i] = shape.dim(i);
  }
  return out;
}

void afToFlDims(const af::dim4& d, Shape& s) {
  auto& storage = s.get();
  if (d.elements() == 0) {
    storage.resize(0);
    return;
  }
  if (d.elements() == 1) {
    storage.resize(1);
    s[0] = 1;
    return;
  }
  // Number of non-trailing-1 dims
  unsigned idx = AF_MAX_DIMS - 1;
  while (d[idx] == 1) {
    --idx;
  }

  storage.resize(idx + 1);
  for (unsigned i = 0; i <= idx; ++i) {
    s[i] = d[i];
  }
}

Shape afToFlDims(const af::dim4& d) {
  Shape s;
  afToFlDims(d, s);
  return s;
}

} // namespace detail
} // namespace fl
