/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fl {
namespace pkg {
namespace speech {
namespace sfx {

class RandomNumberGenerator {
 public:
  explicit RandomNumberGenerator(int seed = 0);

  /// Returns a random integer N such that minVal <= N <= maxVal
  int randInt(int minVal, int maxVal);

  /// Returns a random floating point number in the range [0.0, 1.0).
  float random();

  /// Returns a random floating point number N such that minVal <= N <= maxVal
  float uniform(float minVal, float mx);

  /// Returns a random floating point from a gaussian(normal) distribution
  /// where mu is the mean, and sigma is the standard deviation
  float gaussian(float mean, float sigma);

 private:
  std::mt19937_64 randomEngine_;
  std::uniform_real_distribution<float> uniformDist_;
  std::normal_distribution<float> gaussianDist_;
};

float rootMeanSquare(const std::vector<float>& signal);

float signalToNoiseRatio(
    const std::vector<float>& signal,
    const std::vector<float>& noise);

std::vector<float> genTestSinWave(
    size_t numSamples,
    size_t freq,
    size_t sampleRate,
    float amplitude);

} // namespace sfx
} // namespace speech
} // namespace pkg
} // namespace fl
