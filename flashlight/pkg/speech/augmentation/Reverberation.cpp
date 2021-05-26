/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/augmentation/Reverberation.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace fl {
namespace pkg {
namespace speech {
namespace sfx {

ReverbEcho::ReverbEcho(
    const ReverbEcho::Config& conf,
    unsigned int seed /* = 0 */)
    : conf_(conf), rng_(seed) {}

void ReverbEcho::applyReverb(
    std::vector<float>& source,
    float initial,
    float firstDelay,
    float rt60) {
  size_t length = source.size();
  std::vector<float> reverb(length, 0);
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
        reverb[delay + j] += echo[j] * frac;
      }

      // Add jitter noise for the attenuation
      jitter = 1 + rng_.uniform(-conf_.jitter_, conf_.jitter_);
      const float attenuation = std::pow(10, -3 * jitter * firstDelay / rt60);

      frac *= attenuation;
    }
  }
  for (int i = 0; i < length; ++i) {
    source[i] += reverb[i];
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

  applyReverb(sound, initial, firstDelay, rt60);
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

} // namespace sfx
} // namespace speech
} // namespace pkg
} // namespace fl
