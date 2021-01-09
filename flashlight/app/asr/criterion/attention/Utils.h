/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/app/asr/criterion/attention/Utils.h"

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace app {
namespace asr {

// to avoid nans when apply log to these var
// which cannot be propagated correctly if we set -inf
constexpr float kAttentionMaskValue = -10000;

fl::Variable maskAttention(
    const fl::Variable& input,
    const fl::Variable& sizes);
} // namespace asr
} // namespace app
} // namespace fl
