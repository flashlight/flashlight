/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace pkg {
namespace speech {

fl::Variable maskAttention(
    const fl::Variable& input,
    const fl::Variable& sizes);
} // namespace speech
} // namespace pkg
} // namespace fl
