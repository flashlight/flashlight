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

#include "flashlight/app/asr/augmentation/AdditiveNoise.h"
#include "flashlight/app/asr/augmentation/Reverberation.h"
#include "flashlight/app/asr/augmentation/SoundEffect.h"
#include "flashlight/app/asr/augmentation/TimeStretch.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

// Values for SoundEffectConfig.type_
constexpr const char* const kAdditiveNoise = "AdditiveNoise";
constexpr const char* const kAmplify = "Amplify";
constexpr const char* const kClampAmplitude = "ClampAmplitude";
constexpr const char* const kNormalize = "Normalize";
constexpr const char* const kReverbEcho = "ReverbEcho";
constexpr const char* const kTimeStretch = "TimeStretch";

struct SoundEffectConfig {
  std::string type_;
  // The fields below should be treated like a union, that is, only the field
  // that corresponds to the type_ field should be used. Union cannot be used
  // here since it does not support types like string.
  bool normalizeOnlyIfTooHigh_ = true;
  AdditiveNoise::Config additiveNoiseConfig_;
  Amplify::Config amplifyConfig_;
  ReverbEcho::Config reverbEchoConfig_;
  TimeStretch::Config timeStretchConfig_;
};

std::shared_ptr<SoundEffect> createSoundEffect(
    const std::vector<SoundEffectConfig>& config,
    unsigned int seed = 0);

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
