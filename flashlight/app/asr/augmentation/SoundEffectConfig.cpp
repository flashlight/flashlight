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

#include "flashlight/app/asr/augmentation/SoundEffectUtil.h"
#include "flashlight/lib/common/System.h"

using namespace ::fl::app::asr::sfx;
using ::fl::lib::dirCreateRecursive;
using ::fl::lib::dirname;

namespace cereal {

template <class Archive>
void serialize(Archive& ar, Amplify::Config& conf) {
  ar(cereal::make_nvp("ratioMin", conf.ratioMin_),
     cereal::make_nvp("ratioMax", conf.ratioMax_),
     cereal::make_nvp("randomSeed", conf.randomSeed_));
}

template <class Archive>
std::string save_minimal(Archive const&, RandomPolicy const& randomPolicy) {
  return randomPolicyToString(randomPolicy);
}

template <class Archive>
void load_minimal(
    Archive const&,
    RandomPolicy& randomPolicy,
    std::string const& str) {
  randomPolicy = stringToRandomPolicy(str);
}

template <class Archive>
void serialize(Archive& ar, AdditiveNoise::Config& conf) {
  ar(cereal::make_nvp("proba", conf.proba_),
     cereal::make_nvp("ratio", conf.ratio_),
     cereal::make_nvp("minSnr", conf.minSnr_),
     cereal::make_nvp("maxSnr", conf.maxSnr_),
     cereal::make_nvp("nClipsMin", conf.nClipsMin_),
     cereal::make_nvp("nClipsMax", conf.nClipsMax_),
     cereal::make_nvp("listFilePath", conf.listFilePath_),
     cereal::make_nvp("dsetRndPolicy", conf.dsetRndPolicy_),
     cereal::make_nvp("randomSeed", conf.randomSeed_));
}

template <class Archive>
void serialize(Archive& ar, ReverbEcho::Config& conf) {
  ar(cereal::make_nvp("proba", conf.proba_),
     cereal::make_nvp("initialMin", conf.initialMin_),
     cereal::make_nvp("initialMax", conf.initialMax_),
     cereal::make_nvp("rt60Min", conf.rt60Min_),
     cereal::make_nvp("rt60Max", conf.rt60Max_),
     cereal::make_nvp("firstDelayMin", conf.firstDelayMin_),
     cereal::make_nvp("firstDelayMax", conf.firstDelayMax_),
     cereal::make_nvp("repeat", conf.repeat_),
     cereal::make_nvp("jitter", conf.jitter_),
     cereal::make_nvp("sampleRate", conf.sampleRate_),
     cereal::make_nvp("randomSeed", conf.randomSeed_));
}

template <class Archive>
void serialize(Archive& ar, TimeStretch::Config& conf) {
  ar(cereal::make_nvp("proba", conf.proba_),
     cereal::make_nvp("factorMin", conf.factorMin_),
     cereal::make_nvp("factorMax", conf.factorMax_),
     cereal::make_nvp("window", conf.window_),
     cereal::make_nvp("shift", conf.shift_),
     cereal::make_nvp("fading", conf.fading_),
     cereal::make_nvp("leaveLengthUnchanged", conf.leaveLengthUnchanged_),
     cereal::make_nvp("sampleRate", conf.sampleRate_),
     cereal::make_nvp("randomSeed", conf.randomSeed_));
}

template <class Archive>
void serialize(Archive& ar, SoundEffectConfig& conf) {
  ar(cereal::make_nvp("type", conf.type_));
  if (conf.type_ == kAdditiveNoise) {
    ar(cereal::make_nvp("additiveNoiseConfig", conf.additiveNoiseConfig_));
  } else if (conf.type_ == kAmplify) {
    ar(cereal::make_nvp("amplifyConfig", conf.amplifyConfig_));
  } else if (conf.type_ == kNormalize) {
    ar(cereal::make_nvp(
        "normalizeOnlyIfTooHigh", conf.normalizeOnlyIfTooHigh_));
  } else if (conf.type_ == kReverbEcho) {
    ar(cereal::make_nvp("reverbEchoConfig", conf.reverbEchoConfig_));
  } else if (conf.type_ == kTimeStretch) {
    ar(cereal::make_nvp("timeStretchConfig", conf.timeStretchConfig_));
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
    } else if (conf.type_ == kTimeStretch) {
      sfxChain->add(std::make_shared<TimeStretch>(conf.timeStretchConfig_));
    }
  }
  return sfxChain;
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
