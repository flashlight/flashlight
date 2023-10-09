/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "flashlight/fl/common/Defines.h"

namespace fl {

namespace detail {

/**
 * Precision specifications for autograd operators based on optimization level.
 */
const std::unordered_map<OptimLevel, std::unordered_set<std::string>>
    kOptimLevelTypeExclusionMappings = {
        {OptimLevel::DEFAULT, {}}, // unused
        {OptimLevel::O1,
         // Perform all operations in fp16 except for:
         {"batchnorm",
          "reciprocal",
          "erf",
          "exp",
          "log",
          "log1p",
          "pow",
          "sum",
          "mean",
          "var",
          "norm",
          "normalize",
          "softmax",
          "logSoftmax",
          "categoricalCrossEntropy",
          "gelu"}},
        {OptimLevel::O2,
         // Perform all operations in fp16 except for:
         {"batchnorm"}},
        {OptimLevel::O3, {}} // Perform all operations in f16
};

} // namespace detail

} // namespace fl
