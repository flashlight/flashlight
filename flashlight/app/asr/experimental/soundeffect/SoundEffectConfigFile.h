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

// Example of sound effect chain configuration serialized to json:
// {
//     "soundEffectChain": [
//         {
//             "type_": "AdditiveNoise",
//             "noiseConfig_": {
//                 "maxTimeRatio_": 1.0,
//                 "minSnr_": 0.0,
//                 "maxSnr_": 10.0,
//                 "nClipsPerUtteranceMin_": 0,
//                 "nClipsPerUtteranceMax_": 1,
//                 "listFilePath_":
//                 "experiments/noise-unbalanced-16kHz-mono-train.lst",
//                 "randomNoiseWithReplacement_": true,
//                 "randomSeed_": 1234
//             }
//         },
//         {
//             "type_": "ReverbEcho",
//             "reverbEchoConfig_": {
//                 "absorptionCoefficientMin_": 0.01,
//                 "absorptionCoefficientMax_": 0.1,
//                 "distanceToWallInMetersMin_": 1.0,
//                 "distanceToWallInMetersMax_": 10.0,
//                 "numEchosMin_": 0,
//                 "numEchosMax_": 10,
//                 "jitter_": 0.10000000149011612,
//                 "randomSeed_": 1234567
//             }
//         },
//         {
//             "type_": "ReverbDataset",
//             "reverbDatasetConfig_": {
//                 "listFilePath_":
//                 flashlight/flashlight/app/asr/experimental/soundeffect/test/data/BUT/reverb.lst",
//                 "randomRirWithReplacement_": true,
//                 "randomSeed_": 12
//             }
//         },
//         {
//             "type_": "Amplify",
//             "reverbDatasetConfig_": {
//                 "ratioMin_": 0.1,
//                 "ratioMax_": 0.9,
//                 "randomSeed_": 5489
//             }
//         },
//         {
//             "type_": "Normalize"
//         }
//     ]
// }

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
