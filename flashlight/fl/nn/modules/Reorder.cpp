/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Reorder.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"

namespace fl {

Reorder::Reorder(int dim0, int dim1, int dim2, int dim3)
    : dim0_(dim0), dim1_(dim1), dim2_(dim2), dim3_(dim3) {}

Variable Reorder::forward(const Variable& input) {
  return reorder(input, dim0_, dim1_, dim2_, dim3_);
}

std::string Reorder::prettyString() const {
  std::ostringstream ss;
  ss << "Reorder";
  ss << " (" << dim0_ << "," << dim1_ << "," << dim2_ << "," << dim3_ << ")";
  return ss.str();
}

} // namespace fl
