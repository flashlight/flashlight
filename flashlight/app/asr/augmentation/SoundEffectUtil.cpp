/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/augmentation/SoundEffectUtil.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include <glog/logging.h>

#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/lib/common/System.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

using ::fl::app::asr::saveSound;
using ::fl::lib::dirCreateRecursive;
using ::fl::lib::getTmpPath;
using ::fl::lib::pathsConcat;

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

void scale(std::vector<float>& signal, float scaleFactor) {
  std::transform(
      signal.begin(),
      signal.end(),
      signal.begin(),
      [scaleFactor](float f) -> float { return f * scaleFactor; });
}

float dotProduct(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int size) {
  if (size < 0) {
    size = std::min(a.size(), b.size());
  }
  float ret = 0;
  for (int i = 0; i < size; ++i) {
    ret += a[i] * b[i];
  }
  return ret;
}

std::vector<std::string> loadListFile(
    const std::string& filename,
    const std::string& msg) {
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error(msg + " failed to open list file=" + filename);
  }
  std::vector<std::string> result;
  while (!file.eof()) {
    try {
      std::string filename;
      std::getline(file, filename);
      if (!filename.empty()) {
        result.push_back(filename);
      }
    } catch (std::exception& ex) {
      throw std::runtime_error(
          msg + " failed to read list file=" + filename +
          " with error=" + ex.what());
    }
  }
  if (result.empty()) {
    throw std::runtime_error(msg + " list file=" + filename + " is empty.");
  }
  return result;
}

std::string createTestListFile(
    const std::string& dirName,
    const std::string& basename,
    const std::vector<std::vector<float>> sounds) {
  const std::string tmpDir = getTmpPath(dirName);
  dirCreateRecursive(tmpDir);
  const std::string listFilePath = pathsConcat(tmpDir, basename + ".lst");
  std::ofstream listFile(listFilePath);

  for (int i = 0; i < sounds.size(); ++i) {
    std::stringstream ss;
    ss << basename << '-' << i << ".flac";
    const std::string soundFilePath = pathsConcat(tmpDir, ss.str());
    saveSound(
        soundFilePath,
        sounds[i],
        testSampleRate,
        1,
        fl::app::asr::SoundFormat::FLAC,
        fl::app::asr::SoundSubFormat::PCM_16);
    listFile << soundFilePath << std::endl;
  }
  LOG(INFO) << "created list file=" << listFilePath;
  return listFilePath;
}

std::vector<float> createTestImpulseResponse(size_t size) {
  const float firstDelay = 0.0001;
  const float rt60 = (float)size / (float)testSampleRate;
  RandomNumberGenerator rng;
  std::vector<float> rir(size, 0);
  float frac = 1;
  for (int i = 0; i < rir.size(); ++i) {
    float jitter = 1 + rng.uniform(-0.1, 0.1);
    const float attenuation = std::pow(10, -3 * jitter * firstDelay / rt60);
    frac *= attenuation;
    rir[i] = frac;
  }
  return rir;
}

std::vector<float> genTestSinWave(size_t size, size_t freq, float amplitude) {
  std::vector<float> output(size, 0);
  const float waveLenSamples =
      static_cast<float>(testSampleRate) / static_cast<float>(freq);
  const float ratio = (2 * M_PI) / waveLenSamples;

  for (size_t i = 0; i < size; ++i) {
    output.at(i) = amplitude * std::sin(static_cast<float>(i) * ratio);
  }
  return output;
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
