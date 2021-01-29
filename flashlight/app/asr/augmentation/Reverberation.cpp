/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/augmentation/Reverberation.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include <glog/logging.h>

#include "flashlight/app/asr/augmentation/SoundEffectUtil.h"
#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/lib/mkl/Conv.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

ReverbEcho::ReverbEcho(
    const ReverbEcho::Config& conf,
    unsigned int seed /* = 0 */)
    : conf_(conf), rng_(seed) {}

void ReverbEcho::reverb(
    std::vector<float>& source,
    float initial,
    float firstDelay,
    float rt60) {
  size_t length = source.size();
  for (int i = 0; i < conf_.repeat_; ++i) {
    float frac = 1;
    // echo = initial * source
    std::vector<float> echo = source;
    std::transform(
        echo.begin(), echo.end(), echo.begin(), [initial](float x) -> float {
          return x * initial;
        });
    while (frac > 1e-3) {
      // Add jitter noise for the delay
      float jitter = 1 + rng_.uniform(-conf_.jitter_, conf_.jitter_);
      size_t delay = 1 + int(jitter * firstDelay * conf_.sampleRate_);
      if (delay > length - 1) {
        break;
      }
      for (int j = 0; j < length - delay - 1; ++j) {
        source[delay + j] += echo[j] * frac;
      }

      // Add jitter noise for the attenuation
      jitter = 1 + rng_.uniform(-conf_.jitter_, conf_.jitter_);
      const float attenuation = std::pow(10, -3 * jitter * firstDelay / rt60);

      frac *= attenuation;
    }
  }
}

void ReverbEcho::apply(std::vector<float>& sound) {
  if (rng_.random() >= conf_.proba_) {
    return;
  }
  // Sample characteristics for the reverb
  float initial = rng_.uniform(conf_.initialMin_, conf_.initialMax_);
  float firstDelay = rng_.uniform(conf_.firstDelayMin_, conf_.firstDelayMax_);
  float rt60 = rng_.uniform(conf_.rt60Min_, conf_.rt60Max_);

  reverb(sound, initial, firstDelay, rt60);
}

std::string ReverbEcho::prettyString() const {
  return "ReverbEcho{conf_=" + conf_.prettyString() + "}}";
}

std::string ReverbEcho::Config::prettyString() const {
  std::stringstream ss;
  ss << " proba_=" << proba_ << " initialMin_=" << initialMin_
     << " initialMax_=" << initialMax_ << " rt60Min_=" << rt60Min_
     << " rt60Max_=" << rt60Max_ << " firstDelayMin_=" << firstDelayMin_
     << " firstDelayMax_=" << firstDelayMax_ << " repeat_=" << repeat_
     << " jitter_=" << jitter_ << " sampleRate_=" << sampleRate_;
  return ss.str();
}

ReverbDataset::ReverbDataset(const ReverbDataset::Config& conf)
    : conf_(conf), rng_(conf.randomSeed_) {
  std::ifstream listFile(conf_.listFilePath_);
  if (!listFile) {
    throw std::runtime_error(
        "ReverbDataset failed to open listFilePath_=" + conf_.listFilePath_);
  }
  while (!listFile.eof()) {
    try {
      std::string filename;
      std::getline(listFile, filename);
      if (!filename.empty()) {
        rirFiles_.push_back(filename);
      }
    } catch (std::exception& ex) {
      throw std::runtime_error(
          "ReverbDataset failed to read listFilePath_=" + conf_.listFilePath_ +
          " with error=" + ex.what());
    }
  }
}

/**
 * Implemeneted algorithm is similar to the one found in
 * https://kaldi-asr.org/doc/wav-reverberate_8cc_source.html
 */
void ReverbDataset::apply(std::vector<float>& signal) {
  auto curRirFileIdx = rng_.randInt(0, rirFiles_.size() - 1);
  auto rir = loadSound<float>(rirFiles_[curRirFileIdx]);
  if (rir.empty() || signal.empty()) {
    return;
  }
  const size_t inputSize = signal.size();
  const size_t outputSize = inputSize + rir.size() - 1;
  const float powerBeforeRevereb =
      dotProduct(signal, signal, inputSize) / inputSize;

  const size_t pad = rir.size();
  std::vector<float> paddedSignal(inputSize + pad * 2, 0);
  for (int i = 0; i < inputSize; ++i) {
    paddedSignal[pad + i] = signal[i];
  }

  auto augmented = fl::lib::mkl::conv1D(rir, paddedSignal);

  // signal = augmented without the left (past) padding.
  signal = std::vector<float>(outputSize, 0);
  for (int i = 0; i < signal.size(); ++i) {
    signal[i] = augmented[pad + i];
  }

  float scaleFactor = conf_.volume_;
  if (scaleFactor <= 0) {
    const float powerAfterRevereb =
        dotProduct(signal, signal, inputSize) / inputSize;
    scaleFactor = sqrt(powerBeforeRevereb / powerAfterRevereb);
  }
  std::transform(
      signal.begin(),
      signal.end(),
      signal.begin(),
      [scaleFactor](float f) -> float { return f * scaleFactor; });
}

std::string ReverbDataset::prettyString() const {
  std::stringstream ss;
  ss << "ReverbDataset{conf_=" << conf_.prettyString()
     << "} rirFiles_.size()=" << rirFiles_.size() << "}";
  return ss.str();
}

std::string ReverbDataset::Config::prettyString() const {
  std::stringstream ss;
  ss << "ReverbDataset::Config{proba_=" << proba_
     << " listFilePath_=" << listFilePath_ << " volume_" << volume_
     << " randomSeed_=" << randomSeed_ << '}';
  return ss.str();
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
