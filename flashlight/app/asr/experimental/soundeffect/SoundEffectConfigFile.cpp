/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/experimental/soundeffect/SoundEffectConfigFile.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

#include "flashlight/app/asr/experimental/soundeffect/Sound.h"
#include "flashlight/lib/common/System.h"

using ::fl::lib::dirCreateRecursive;
using ::fl::lib::dirname;
using namespace ::fl::app::asr::sfx;

namespace cereal {

template <class Archive>
void serialize(Archive& ar, AdditiveNoise::Config& conf) {
  ar(cereal::make_nvp("maxTimeRatio_", conf.maxTimeRatio_),
     cereal::make_nvp("minSnr_", conf.minSnr_),
     cereal::make_nvp("maxSnr_", conf.maxSnr_),
     cereal::make_nvp("nClipsPerUtteranceMin_", conf.nClipsPerUtteranceMin_),
     cereal::make_nvp("nClipsPerUtteranceMax_", conf.nClipsPerUtteranceMax_),
     cereal::make_nvp("listFilePath_", conf.listFilePath_),
     cereal::make_nvp(
         "randomNoiseWithReplacement_", conf.randomNoiseWithReplacement_),
     cereal::make_nvp("randomSeed_", conf.randomSeed_));
}

template <class Archive>
void serialize(Archive& ar, ReverbEcho::Config& conf) {
  ar(cereal::make_nvp(
         "absorptionCoefficientMin_", conf.absorptionCoefficientMin_),
     cereal::make_nvp(
         "absorptionCoefficientMax_", conf.absorptionCoefficientMax_),
     cereal::make_nvp(
         "distanceToWallInMetersMin_", conf.distanceToWallInMetersMin_),
     cereal::make_nvp(
         "distanceToWallInMetersMax_", conf.distanceToWallInMetersMax_),
     cereal::make_nvp("numEchosMin_", conf.numEchosMin_),
     cereal::make_nvp("numEchosMax_", conf.numEchosMax_),
     cereal::make_nvp("jitter_", conf.jitter_),
     cereal::make_nvp("randomSeed_", conf.randomSeed_));
}

template <class Archive>
void serialize(Archive& ar, ReverbDataset::Config& conf) {
  ar(cereal::make_nvp("listFilePath_", conf.listFilePath_),
     cereal::make_nvp(
         "randomRirWithReplacement_", conf.randomRirWithReplacement_),
     cereal::make_nvp("randomSeed_", conf.randomSeed_));
}

template <class Archive>
void serialize(Archive& ar, Amplify::Config& conf) {
  ar(cereal::make_nvp("ratioMin_", conf.ratioMin_),
     cereal::make_nvp("ratioMax_", conf.ratioMax_),
     cereal::make_nvp("randomSeed_", conf.randomSeed_));
}

template <class Archive>
void serialize(Archive& ar, SoundEffectConfig& conf) {
  ar(cereal::make_nvp("type_", conf.type_));
  if (conf.type_ == kAdditiveNoise) {
    ar(cereal::make_nvp("noiseConfig_", conf.noiseConfig_));
  } else if (conf.type_ == kReverbDataset) {
    ar(cereal::make_nvp("reverbDatasetConfig_", conf.reverbDatasetConfig_));
  } else if (conf.type_ == kReverbEcho) {
    ar(cereal::make_nvp("reverbEchoConfig_", conf.reverbEchoConfig_));
  } else if (conf.type_ == kAmplify) {
    ar(cereal::make_nvp("amplifyConfig_", conf.amplifyConfig_));
  }
}

} // namespace cereal

namespace fl {
namespace app {
namespace asr {
namespace sfx {

void writeSoundEffectConfigFile(
    const std::string& filename,
    const std::vector<SoundEffectConfig>& sfxConfigs) {
  try {
    const std::string path = dirname(filename);
    dirCreateRecursive(path);
    std::ofstream file(filename, std::ofstream::trunc);
    cereal::JSONOutputArchive archive(file);
    archive(cereal::make_nvp("soundEffectChain", sfxConfigs));
    FL_LOG(fl::INFO) << "writeSoundEffectConfigFile(filename=" << filename
                     << ")  sfxConfigs.size()=" << sfxConfigs.size();
  } catch (std::exception& ex) {
    std::stringstream ss;
    ss << "writeSoundEffectConfigFile(filename=" << filename
       << ") failed with error={" << ex.what() << "}";
    throw std::runtime_error(ss.str());
  }
}

std::vector<SoundEffectConfig> readSoundEffectConfigFile(
    const std::string& filename) {
  try {
    std::ifstream file(filename);
    cereal::JSONInputArchive archive(file);
    std::vector<SoundEffectConfig> sfxConfigs;
    archive(sfxConfigs);
    FL_LOG(fl::INFO) << "readSoundEffectConfigFile(filename=" << filename
                     << ") sfxConfigs.size()=" << sfxConfigs.size();
    return sfxConfigs;
  } catch (std::exception& ex) {
    std::stringstream ss;
    ss << "readSoundEffectConfigFile(filename=" << filename
       << ") failed with error={" << ex.what() << "}";
    throw std::runtime_error(ss.str());
  }
}

std::shared_ptr<SoundEffect> createSoundEffect(
    const std::vector<SoundEffectConfig>& sfxConfigs) {
  auto sfxChain = std::make_shared<SoundEffectChain>();
  for (const SoundEffectConfig& conf : sfxConfigs) {
    if (conf.type_ == kAdditiveNoise) {
      sfxChain->add(std::make_shared<AdditiveNoise>(conf.noiseConfig_));
    } else if (conf.type_ == kReverbEcho) {
      sfxChain->add(std::make_shared<ReverbEcho>(conf.reverbEchoConfig_));
    } else if (conf.type_ == kReverbDataset) {
      sfxChain->add(std::make_shared<ReverbDataset>(conf.reverbDatasetConfig_));
    } else if (conf.type_ == kNormalize) {
      sfxChain->add(std::make_shared<Normalize>());
    } else if (conf.type_ == kAmplify) {
      sfxChain->add(std::make_shared<Amplify>(conf.amplifyConfig_));
    }
  }
  return sfxChain;
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
