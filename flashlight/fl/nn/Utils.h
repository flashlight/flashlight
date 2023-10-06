/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * \defgroup nn_utils NN Utils
 * @{
 */

#pragma once

#include <iomanip>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/nn/modules/Module.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/**
 * Compute the total number of parameters of a fl::Module.
 *
 * @param[in] module The module over which to compute params
 * @return the number of parameters in the module
 */
FL_API int64_t numTotalParams(std::shared_ptr<fl::Module> module);

/**
 * Returns true if the parameters of two modules are of same type and are
 * element-wise equal within a given tolerance limit.
 *
 * @param [a,b] input Modules to compare
 * @param absTolerance absolute tolerance allowed
 *
 */
FL_API bool
allParamsClose(const Module& a, const Module& b, double absTolerance = 1e-5);

namespace detail {

FL_API int64_t getNumRnnParams(
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

FL_API int
derivePadding(int inSz, int filterSz, int stride, int pad, int dilation);

/// packs a list of arrays (possibly of different dimensions) to a single array
/// by padding them to same dimensions
FL_API Tensor join(
    const std::vector<Tensor>& inputs,
    double padValue = 0.0,
    int batchDim = -1);

/** @} */

} // namespace fl
