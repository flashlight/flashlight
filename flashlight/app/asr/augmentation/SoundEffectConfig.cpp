/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/augmentation/SoundEffectConfig.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

#include "flashlight/lib/common/System.h"

using namespace ::fl::app::asr::sfx;
using ::fl::lib::dirname;
using ::fl::lib::dirCreateRecursive;

namespace cereal {

template <class Archive>
void serialize(Archive& ar, Amplify::Config& conf) {
  ar(cereal::make_nvp("ratioMin", conf.ratioMin_),
     cereal::make_nvp("ratioMax", conf.ratioMax_),
     cereal::make_nvp("randomSeed", conf.randomSeed_));
}

template <class Archive>
void serialize(Archive& ar, SoundEffectConfig& conf) {
  ar(cereal::make_nvp("type", conf.type_));
  if (conf.type_ == kNormalize) {
    ar(cereal::make_nvp(
        "normalizeOnlyIfTooHigh", conf.normalizeOnlyIfTooHigh_));
  } else if (conf.type_ == kAmplify) {
    ar(cereal::make_nvp("amplifyConfig", conf.amplifyConfig_));
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
    if (conf.type_ == kNormalize) {
      sfxChain->add(std::make_shared<Normalize>(conf.normalizeOnlyIfTooHigh_));
    } else if (conf.type_ == kClampAmplitude) {
      sfxChain->add(std::make_shared<ClampAmplitude>());
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
