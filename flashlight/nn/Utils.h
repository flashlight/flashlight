/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
/**
 * @file nn/Utils.h
 *
 * Utils for modules.
 */

#pragma once

#include <af/dim4.hpp>

#include "flashlight/common/Defines.h"
#include "flashlight/nn/modules/Module.h"

namespace fl {

/**
 * Returns true if the parameters of two modules are of same type and are
 * element-wise equal within given tolerance limit.
 *
 * @param [a,b] input Modules to compare
 * @param absTolerance absolute tolerance allowed
 *
 */
bool allParamsClose(
    const Module& a,
    const Module& b,
    double absTolerance = 1e-5);

namespace detail {

int64_t getNumRnnParams(
    int input_size,
    int hidden_size,
    int num_layers,
    RnnMode mode,
    bool bidirectional);

/// used for Conv2D and Pool2D params
struct IntOrPadMode {
  /* implicit */ IntOrPadMode(int val) : padVal(val) {}
  /* implicit */ IntOrPadMode(PaddingMode mode)
      : padVal(static_cast<int>(mode)) {}
  const int padVal;
};

} // namespace detail

int derivePadding(int inSz, int filterSz, int stride, int pad, int dilation);

int64_t numTotalParams(std::shared_ptr<fl::Module> module);

/// packs a list of arrays (possibly of different dimensions) to a single array
/// by padding them to same dimensions
af::array join(
    const std::vector<af::array>& inputs,
    double padValue = 0.0,
    dim_t batchDim = -1);

} // namespace fl
