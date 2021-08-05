/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * \defgroup nn_utils NN Utils
 * @{
 */

#pragma once

#include <iomanip>

#include <af/dim4.hpp>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/**
 * Compute the total number of parameters of a fl::Module.
 *
 * @param[in] module The module over which to compute params
 * @return the number of parameters in the module
 */
int64_t numTotalParams(std::shared_ptr<fl::Module> module);

/**
 * Returns true if the parameters of two modules are of same type and are
 * element-wise equal within a given tolerance limit.
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

/// packs a list of arrays (possibly of different dimensions) to a single array
/// by padding them to same dimensions
af::array join(
    const std::vector<af::array>& inputs,
    double padValue = 0.0,
    dim_t batchDim = -1);

/** @} */

} // namespace fl
