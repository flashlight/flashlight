/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/augmentation/GaussianNoise.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace fl {
namespace app {
namespace asr {
namespace sfx {

std::string GaussianNoise::Config::prettyString() const {
  std::stringstream ss;
  ss << "GaussianNoise::Config{minSnr_=" << minSnr_ << " maxSnr_=" << maxSnr_
     << '}';
  return ss.str();
}

std::string GaussianNoise::prettyString() const {
  std::stringstream ss;
  ss << "GaussianNoise{config={" << conf_.prettyString() << "}}";
  return ss.str();
};

GaussianNoise::GaussianNoise(
    const GaussianNoise::Config& config,
    unsigned int seed /* = 0 */)
    : conf_(config), rng_(seed) {}

void GaussianNoise::apply(std::vector<float>& signal) {
  if (rng_.random() >= conf_.proba_) {
    return;
  }
  const float signalRms = rootMeanSquare(signal);
  const float snr = rng_.uniform(conf_.minSnr_, conf_.maxSnr_);
  const float noiseMult = signalRms / std::pow(10, snr / 20.0);

  for (int i = 0; i < signal.size(); ++i) {
    signal[i] += rng_.gaussian(0, noiseMult);
  }
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
