/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace app {
namespace asr {

// attention
const std::string kContentAttention = "content";
const std::string kKeyValueAttention = "keyvalue";
const std::string kLocationAttention = "location";
const std::string kMultiHeadContentAttention = "multi";
const std::string kMultiHeadKeyValueContentAttention = "multikv";
const std::string kMultiHeadSplitContentAttention = "multisplit";
const std::string kMultiHeadKeyValueSplitContentAttention = "multikvsplit";
const std::string kNeuralContentAttention = "neural";
const std::string kNeuralLocationAttention = "neuralloc";
const std::string kSimpleLocationAttention = "simpleloc";

// window
const std::string kMedianWindow = "median";
const std::string kNoWindow = "no";
const std::string kSoftWindow = "soft";
const std::string kSoftPretrainWindow = "softPretrain";
const std::string kStepWindow = "step";

// to avoid nans when apply log to these var
// which cannot be propagated correctly if we set -inf
constexpr float kAttentionMaskValue = -10000;

} // namespace asr
} // namespace app
} // namespace fl
