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
namespace app {
namespace asr {
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

void scale(std::vector<float>& signal, float scaleFactor);

/**
 * Returns the dot product of the first size elements of a and b vectors.
 * when size < 0 size is set to be min(size(a), size(b))
 */
float dotProduct(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int size = -1);

std::vector<std::string> loadListFile(
    const std::string& filename,
    const std::string& msg = "");

/**
 * testing support utilities are below.
 */
constexpr size_t testSampleRate = 16000;

/**
 * return the fullpath to generated list file containing the given sounds.
 */
std::string createTestListFile(
    const std::string& dirName,
    const std::string& basename,
    const std::vector<std::vector<float>> sounds);

/**
 *  Generate a random RIR with exponential decay.
 */
std::vector<float> createTestImpulseResponse(size_t size);

std::vector<float> genTestSinWave(size_t size, size_t freq, float amplitude);

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
