/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/augmentation/SoundEffectUtil.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace fl {
namespace app {
namespace asr {
namespace sfx {

RandomNumberGenerator::RandomNumberGenerator(int seed /* = 0 */)
    : randomEngine_(seed), uniformDist_(0, 1), gaussianDist_(0, 1) {}

int RandomNumberGenerator::randInt(int minVal, int maxVal) {
  if (minVal > maxVal) {
    std::swap(minVal, maxVal);
  }
  return randomEngine_() % (maxVal - minVal + 1) + minVal;
}

float RandomNumberGenerator::random() {
  return uniformDist_(randomEngine_);
}

float RandomNumberGenerator::uniform(float minVal, float maxVal) {
  return minVal + (maxVal - minVal) * uniformDist_(randomEngine_);
}

float RandomNumberGenerator::gaussian(float mean, float sigma) {
  return mean + gaussianDist_(randomEngine_) * sigma;
}

float rootMeanSquare(const std::vector<float>& signal) {
  float sumSquares = 0;
  for (int i = 0; i < signal.size(); ++i) {
    sumSquares += signal[i] * signal[i];
  }
  return std::sqrt(sumSquares / signal.size());
}

float signalToNoiseRatio(
    const std::vector<float>& signal,
    const std::vector<float>& noise) {
  auto singalRms = rootMeanSquare(signal);
  auto noiseRms = rootMeanSquare(noise);
  return 20 * std::log10(singalRms / noiseRms);
}

std::vector<float>
genTestSinWave(size_t numSamples, size_t freq, size_t sampleRate, float amplitude) {
  std::vector<float> output(numSamples, 0);
  const float waveLenSamples =
      static_cast<float>(sampleRate) / static_cast<float>(freq);
  const float ratio = (2 * M_PI) / waveLenSamples;

  for (size_t i = 0; i < numSamples; ++i) {
    output.at(i) = amplitude * std::sin(static_cast<float>(i) * ratio);
  }
  return output;
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
