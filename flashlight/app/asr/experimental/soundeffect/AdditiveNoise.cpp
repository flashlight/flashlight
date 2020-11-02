/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/experimental/soundeffect/AdditiveNoise.h"

#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

#include "flashlight/app/asr/experimental/soundeffect/ListFile.h"
#include "flashlight/fl/common/Logging.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

std::string AdditiveNoise::Config::prettyString() const {
  std::stringstream ss;
  ss << "maxTimeRatio_=" << maxTimeRatio_ << " minSnr_=" << minSnr_
     << " maxSnr_=" << maxSnr_
     << " nClipsPerUtteranceMin_=" << nClipsPerUtteranceMin_
     << " nClipsPerUtteranceMax_" << nClipsPerUtteranceMax_
     << " listFilePath_=" << listFilePath_;
  return ss.str();
}

std::string AdditiveNoise::prettyString() const {
  std::stringstream ss;
  ss << "config={" << conf_.prettyString() << '}' << "soundLoader_={"
     << (soundLoader_ ? soundLoader_->prettyString() : "null") << '}'
     << " SoundEffect={" << SoundEffect::prettyString() << "}";
  return ss.str();
};

Interval AdditiveNoise::augmentationInterval(size_t size) {
  const int numAugSignalSamples =
      static_cast<double>(size) * conf_.maxTimeRatio_;

  if (numAugSignalSamples <= 0) {
    return {0UL, 0UL};
  } else if (numAugSignalSamples == size) {
    return {0UL, numAugSignalSamples};
  } else {
    const int augStart =
        (uniformDistribution_(randomEngine_) % (size - numAugSignalSamples));
    return {augStart, augStart + numAugSignalSamples};
  }
}

std::vector<Sound> AdditiveNoise::loadNoises() {
  int nClipsPerUtterance = 0;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!soundLoader_) {
      if (conf_.randomNoiseWithReplacement_) {
        soundLoader_ = std::make_shared<SoundLoaderRandomWithReplacement>(
            conf_.listFilePath_, conf_.randomSeed_);
      } else {
        soundLoader_ = std::make_shared<SoundLoaderRandomWithoutReplacement>(
            conf_.listFilePath_, conf_.randomSeed_);
      }
    }
    nClipsPerUtterance = randomNumClipsPerUtterance_(randomEngine_);
  }

  std::vector<Sound> noiseSounds;
  for (int i = 0; i < nClipsPerUtterance; ++i) {
    try {
      Sound sound = soundLoader_->loadRandom();
      if (!sound.empty()) {
        noiseSounds.push_back(sound);
      }
    } catch (const std::exception& ex) {
      std::cerr << "AdditiveNoise::loadNoises() failed with error="
                << ex.what();
    }
  }
  return noiseSounds;
}

AdditiveNoise::AdditiveNoise(const AdditiveNoise::Config config)
    : conf_(config),
      randomEngine_(config.randomSeed_),
      uniformDistribution_(0, std::numeric_limits<int>::max()),
      randomNumClipsPerUtterance_(
          config.nClipsPerUtteranceMin_,
          config.nClipsPerUtteranceMax_),
      randomSnr_(config.minSnr_, config.maxSnr_) {}

Sound AdditiveNoise::getNoise(size_t size) {
  const Interval augInterval = augmentationInterval(size);
  if (intervalSize(augInterval) <= 0) {
    return {};
  }
  std::vector<Sound> noiseSounds = loadNoises();
  if (noiseSounds.empty()) {
    return {};
  }

  auto noiseData = std::make_shared<std::vector<float>>(size, 0.0f);
  for (int i = 0; i < noiseSounds.size(); ++i) {
    std::shared_ptr<std::vector<float>> curNoiseData =
        noiseSounds[i].getCpuData();

    const int noiseShift =
        uniformDistribution_(randomEngine_) % curNoiseData->size();

    for (int j = 0; j < intervalSize(augInterval); ++j) {
      noiseData->at(augInterval.first + j) =
          curNoiseData->at((noiseShift + j) % curNoiseData->size());
    }
  }
  return Sound(noiseData);
}

Sound AdditiveNoise::applyImpl(Sound signal) {
  Sound noise = getNoise(signal.size());
  if (!noise.empty()) {
    try {
      signal.addNoise(noise, randomSnr_(randomEngine_));
    } catch (std::exception& ex) {
      FL_LOG(fl::WARNING) << "AdditiveNoise::applyImpl(signal={"
                          << signal.prettyString() << "}) noise={"
                          << noise.prettyString() << "} failed with error={"
                          << ex.what() << "}";
    }
  }
  return signal;
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
