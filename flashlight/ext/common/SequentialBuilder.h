/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/contrib/contrib.h"
#include "flashlight/fl/flashlight.h"

namespace fl {
namespace ext {

/**
 * Build a sequential module by parsing a file that
 * defines the model architecture.
 */
std::shared_ptr<fl::Sequential> buildSequentialModule(
    const std::string& archfile,
    int64_t nFeatures,
    int64_t nClasses);
} // namespace ext
} // namespace fl
