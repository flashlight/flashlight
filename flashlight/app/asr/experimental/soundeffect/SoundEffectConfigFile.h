/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <flashlight/fl/flashlight.h>

#include "flashlight/app/asr/experimental/soundeffect/AdditiveNoise.h"
#include "flashlight/app/asr/experimental/soundeffect/Reverberation.h"
#include "flashlight/app/asr/experimental/soundeffect/SoundEffect.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

// Values for SoundEffectConfig.type_
constexpr const char* const kAdditiveNoise = "AdditiveNoise";
constexpr const char* const kReverbEcho = "ReverbEcho";
constexpr const char* const kReverbDataset = "ReverbDataset";
constexpr const char* const kNormalize = "Normalize";
constexpr const char* const kAmplify = "Amplify";

// Union of all possible configurations types.
// Instance of this class shall only use the configuration field for
// by the sound effect type_ specified.
struct SoundEffectConfig {
  std::string type_;
  Amplify::Config amplifyConfig_; // use when type_ == kAmplify
  AdditiveNoise::Config noiseConfig_; // use when type_ == kAdditiveNoise
  ReverbEcho::Config reverbEchoConfig_; // use when type_ == kReverbEcho
  ReverbDataset::Config
      reverbDatasetConfig_; // use when type_ == kReverbDataset
};

// Write configuration vector into json file
void writeSoundEffectConfigFile(
    const std::string& filename,
    const std::vector<SoundEffectConfig>& config);

// Read configuration vector from json file
std::vector<SoundEffectConfig> readSoundEffectConfigFile(
    const std::string& filename);

std::shared_ptr<SoundEffect> createSoundEffect(
    const std::vector<SoundEffectConfig>& config);

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
