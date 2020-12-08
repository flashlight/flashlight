/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/augmentation/AdditiveNoise.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include "flashlight/app/asr/data/Sound.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

std::string AdditiveNoise::Config::prettyString() const {
  std::stringstream ss;
  ss << "AdditiveNoise::Config{ratio=" << ratio << " minSnr_=" << minSnr_
     << " maxSnr_=" << maxSnr_ << " nClipsMin_=" << nClipsMin_ << " nClipsMax_"
     << nClipsMax_ << " listFilePath_=" << listFilePath_
     << " dsetRndPolicy_=" << randomPolicyToString(dsetRndPolicy_)
     << " randomSeed_=" << randomSeed_ << '}';
  return ss.str();
}

std::string AdditiveNoise::prettyString() const {
  std::stringstream ss;
  ss << "AdditiveNoise{config={" << conf_.prettyString() << '}'
     << " datasetRandomiser_={" << datasetRandomiser_->prettyString() << "}";
  return ss.str();
};

AdditiveNoise::AdditiveNoise(const AdditiveNoise::Config& config)
    : conf_(config),
      randomEngine_(config.randomSeed_),
      uniformDistribution_(0, std::numeric_limits<int>::max()),
      randomNumClipsPerUtterance_(config.nClipsMin_, config.nClipsMax_),
      randomSnr_(config.minSnr_, config.maxSnr_) {
  std::ifstream listFile(conf_.listFilePath_);
  if (!listFile) {
    throw std::runtime_error(
        "AdditiveNoise failed to open listFilePath_=" + conf_.listFilePath_);
  }
  std::vector<std::string> noiseFiles;
  while (!listFile.eof()) {
    try {
      std::string filename;
      std::getline(listFile, filename);
      if (!filename.empty()) {
        noiseFiles.push_back(filename);
      }
    } catch (std::exception& ex) {
      throw std::runtime_error(
          "AdditiveNoise failed to read listFilePath_=" + conf_.listFilePath_ +
          " with error=" + ex.what());
    }
  }

  DatasetRandomiser<std::string>::Config dsConf;
  dsConf.policy_ = conf_.dsetRndPolicy_;
  dsConf.randomSeed_ = conf_.randomSeed_;
  datasetRandomiser_ = std::make_unique<DatasetRandomiser<std::string>>(
      dsConf, std::move(noiseFiles));
}

namespace {

float rootMeanSquare(const std::vector<float>& signal) {
  float sumOfSquares = 0;
  for (float i : signal) {
    sumOfSquares += i * i;
  }
  return std::sqrt(sumOfSquares / static_cast<float>(signal.size()));
}

// Interval range is like that of a C loop. Inclusive of first and exclusive of
// second. {0,0}, {10,10} are empty intervals.
using Interval = std::pair<int, int>;
int intervalSize(const Interval& interval) {
  return interval.second - interval.first;
}

} // namespace

void AdditiveNoise::apply(std::vector<float>& signal) {
  const float signalRms = rootMeanSquare(signal);
  const float snr = randomSnr_(randomEngine_);

  // Generate random augmentation interval
  Interval augInterval;
  const int numAugSignalSamples =
      static_cast<double>(signal.size()) * conf_.ratio;
  if (numAugSignalSamples <= 0) {
    return;
  } else if (numAugSignalSamples == signal.size()) {
    augInterval = {0UL, numAugSignalSamples};
  } else {
    const int augStart =
        (uniformDistribution_(randomEngine_) %
         (signal.size() - numAugSignalSamples));
    augInterval = {augStart, augStart + numAugSignalSamples};
  }

  // Load random number of noise clips.
  std::vector<std::vector<float>> noiseClips;
  const int nClips = randomNumClipsPerUtterance_(randomEngine_);
  while (noiseClips.size() < nClips) {
    noiseClips.push_back(loadSound<float>(datasetRandomiser_->getRandom()));
  }

  // Sum the noise clips on the augmentation interval.
  std::vector<float> mixedNoise(signal.size(), 0.0f);
  for (const std::vector<float>& curNoise : noiseClips) {
    int noiseShift = uniformDistribution_(randomEngine_) % curNoise.size();

    for (int j = 0; j < intervalSize(augInterval); ++j) {
      // tile the noise if shorter then augInterval.
      mixedNoise.at(augInterval.first + j) =
          curNoise.at((noiseShift + j) % curNoise.size());
    }
  }

  const float noiseRms = rootMeanSquare(mixedNoise);
  if (noiseRms > 0) {
    // https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    const float noiseMult =
        (signalRms / (noiseRms * std::sqrt(std::pow(10, snr / 20.0))));
    for (int i = 0; i < signal.size(); ++i) {
      signal.at(i) += mixedNoise.at(i) * noiseMult;
    }
  } else {
    std::cerr << "AdditiveNoise::apply() invalid noiseRms=" << noiseRms;
  }
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
