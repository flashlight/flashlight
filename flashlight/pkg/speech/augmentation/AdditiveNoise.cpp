/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/augmentation/AdditiveNoise.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include "flashlight/pkg/speech/augmentation/SoundEffectUtil.h"
#include "flashlight/pkg/speech/data/Sound.h"
#include "flashlight/fl/common/Logging.h"

namespace fl {
namespace pkg {
namespace speech {
namespace sfx {

std::string AdditiveNoise::Config::prettyString() const {
  std::stringstream ss;
  ss << "AdditiveNoise::Config{ratio_=" << ratio_ << " minSnr_=" << minSnr_
     << " maxSnr_=" << maxSnr_ << " nClipsMin_=" << nClipsMin_ << " nClipsMax_"
     << nClipsMax_ << " listFilePath_=" << listFilePath_ << '}';
  return ss.str();
}

std::string AdditiveNoise::prettyString() const {
  std::stringstream ss;
  ss << "AdditiveNoise{config={"  << conf_.prettyString() << '}';
  return ss.str();
};

AdditiveNoise::AdditiveNoise(
    const AdditiveNoise::Config& config,
    unsigned int seed /* = 0 */)
    : conf_(config), rng_(seed) {
  std::ifstream listFile(conf_.listFilePath_);
  if (!listFile) {
    throw std::runtime_error(
        "AdditiveNoise failed to open listFilePath_=" + conf_.listFilePath_);
  }
  while (!listFile.eof()) {
    try {
      std::string filename;
      std::getline(listFile, filename);
      if (!filename.empty()) {
        noiseFiles_.push_back(filename);
      }
    } catch (std::exception& ex) {
      throw std::runtime_error(
          "AdditiveNoise failed to read listFilePath_=" + conf_.listFilePath_ +
          " with error=" + ex.what());
    }
  }
}

void AdditiveNoise::apply(std::vector<float>& signal) {
  if (rng_.random() >= conf_.proba_) {
    return;
  }
  const float signalRms = rootMeanSquare(signal);
  const float snr = rng_.uniform(conf_.minSnr_, conf_.maxSnr_);
  const int nClips = rng_.randInt(conf_.nClipsMin_, conf_.nClipsMax_);
  if (nClips == 0) {
    return;
  }
  int augStart = rng_.randInt(0, signal.size() - 1);
  // overflow implies we start at the beginning again.
  int augEnd = augStart + conf_.ratio_ * signal.size();

  std::vector<float> mixedNoise(signal.size(), 0.0f);
  for (int i = 0; i < nClips; ++i) {
    auto curNoiseFileIdx = rng_.randInt(0, noiseFiles_.size() - 1);
    auto curNoise = loadSound<float>(noiseFiles_[curNoiseFileIdx]);
    int shift = rng_.randInt(0, curNoise.size() - 1);
    for (int j = augStart; j < augEnd; ++j) {
      mixedNoise[j % mixedNoise.size()] +=
          curNoise[(shift + j) % curNoise.size()];
    }
  }

  const float noiseRms = rootMeanSquare(mixedNoise);
  if (noiseRms > 0) {
    // https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    const float noiseMult = (signalRms / (noiseRms * std::pow(10, snr / 20.0)));
    for (int i = 0; i < signal.size(); ++i) {
      signal[i] += mixedNoise[i] * noiseMult;
    }
  } else {
    FL_LOG(fl::WARNING) << "AdditiveNoise::apply() invalid noiseRms="
                        << noiseRms;
  }
}

} // namespace sfx
} // namespace speech
} // namespace pkg
} // namespace fl
