/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/integrations/halide/HalideInterface.h"

#include <af/dim4.hpp>

namespace fl {
namespace ext {

/**
 * Gets Halide dims from an ArrayFire array. Halide is column major, so reverse
 * all dimensions.
 */
std::vector<int> afToHalideDims(const af::dim4& dims) {
  const auto ndims = dims.ndims();
  std::vector<int> halideDims(ndims);
  for (int i = 0; i < ndims; ++i) {
    halideDims[ndims - 1 - i] = static_cast<int>(dims.dims[i]);
  }
  return halideDims;
}

/**
 * Gets Halide dims from an ArrayFire array. Halide is column major, so reverse
 * all dimensions.
 */
af::dim4 halideToAfDims(const Halide::Buffer<void>& buffer) {
  const int nDims = buffer.dimensions();
  if (nDims > 4) {
    throw std::invalid_argument(
        "getDims: Halide buffer has greater than 4 dimensions");
  }
  af::dim4 out(1, 1, 1, 1); // initialize so unfilled dims are 1, not 0
  for (size_t i = 0; i < nDims; ++i) {
    // Halide can have size zero along a dim --> convert to size 1 for AF
    auto size = static_cast<dim_t>(buffer.dim(i).extent());
    out[nDims - 1 - i] = size == 0 ? 1 : size;
  }
  return out;
}

af::dtype halideRuntimeTypeToAfType(halide_type_t type) {
  halide_type_code_t typeCode = type.code;
  switch (typeCode) {
    case halide_type_int:
      return af::dtype::s32;
    case halide_type_uint:
      return af::dtype::u32;
    case halide_type_float:
      return af::dtype::f32;
    default:
      throw std::invalid_argument(
          "halideRuntimeTypeToAfType: unsupported or unknown Halide type");
  }
}
}
}
