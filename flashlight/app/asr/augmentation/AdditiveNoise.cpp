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
#include "flashlight/fl/common/Logging.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

std::string AdditiveNoise::Config::prettyString() const {
  std::stringstream ss;
  ss << "AdditiveNoise::Config{ratio_=" << ratio_ << " minSnr_=" << minSnr_
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
      randomProba_(0, 1),
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
  float meanSquares = 0;
  for (int i = 0; i < signal.size(); ++i) {
    meanSquares = (meanSquares * i + signal[i] * signal[i]) / (i + 1);
  }
  return std::sqrt(meanSquares);
}

} // namespace

void AdditiveNoise::apply(std::vector<float>& signal) {
  if (randomProba_(randomEngine_) >= conf_.proba_) {
    return;
  }
  const float signalRms = rootMeanSquare(signal);
  const float snr = randomSnr_(randomEngine_);
  const int nClips = randomNumClipsPerUtterance_(randomEngine_);
  int augStart = uniformDistribution_(randomEngine_) % (signal.size() - 1);
  // overflow implies we start at the beginning again.
  int augEnd = augStart + conf_.ratio_ * signal.size();

  std::vector<float> mixedNoise(signal.size(), 0.0f);
  for (int i = 0; i < nClips; ++i) {
    auto curNoise = loadSound<float>(datasetRandomiser_->getRandom());
    int shift = uniformDistribution_(randomEngine_) % (curNoise.size() - 1);
    for (int j = augStart; j < augEnd; ++j) {
      mixedNoise[j % mixedNoise.size()] +=
          curNoise[(shift + j) % curNoise.size()];
    }
  }

  const float noiseRms = rootMeanSquare(mixedNoise);
  if (noiseRms > 0) {
    // https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    const float noiseMult =
        (signalRms / (noiseRms * std::sqrt(std::pow(10, snr / 20.0))));
    for (int i = 0; i < signal.size(); ++i) {
      signal[i] += mixedNoise[i] * noiseMult;
    }
  } else {
    FL_LOG(fl::WARNING) << "AdditiveNoise::apply() invalid noiseRms="
                        << noiseRms;
  }
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
