/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "flashlight/app/asr/augmentation/SoundEffect.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

// Values for SoundEffectConfig.type_
constexpr const char* const kNormalize = "Normalize";
constexpr const char* const kClampAmplitude = "ClampAmplitude";
constexpr const char* const kAmplify = "Amplify";

struct SoundEffectConfig {
  std::string type_;
  union {
    bool normalizeOnlyIfTooHigh_ = true;
    Amplify::Config amplifyConfig_;
  };
};

std::shared_ptr<SoundEffect> createSoundEffect(
    const std::vector<SoundEffectConfig>& config);

// Write configuration vector into json file
void writeSoundEffectConfigFile(
    const std::string& filename,
    const std::vector<SoundEffectConfig>& config);

// Read configuration vector from json file
std::vector<SoundEffectConfig> readSoundEffectConfigFile(
    const std::string& filename);

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
