/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/pkg/speech/augmentation/AdditiveNoise.h"
#include "flashlight/pkg/speech/augmentation/SoundEffectUtil.h"
#include "flashlight/pkg/speech/data/Sound.h"
#include "flashlight/fl/common/Init.h"
#include "flashlight/lib/common/System.h"

using namespace ::fl::pkg::speech::sfx;
using ::fl::pkg::speech::saveSound;
using ::fl::lib::dirCreateRecursive;
using ::fl::lib::getTmpPath;
using ::fl::lib::pathsConcat;
using ::testing::Pointwise;

const size_t sampleRate = 16000;

MATCHER_P(FloatNearPointwise, tol, "Out of range") {
  return (
      std::get<0>(arg) > std::get<1>(arg) - tol &&
      std::get<0>(arg) < std::get<1>(arg) + tol);
}

/**
 * Test that noise is applied with correct SNR. The test generates signal and
 * noise such that RMS of both is 1. The noise application ratio is set to cover
 * the entire signal. All random ranges are set to zero (min random ==max
 * random) for deterministic behaviour. After augmentation, the augmentation
 * noise is extracted by subtracting the original sound from the augmented
 * found. The test ensure that the extracted noise matches the original noise
 * considering the SNR value.
 */
TEST(AdditiveNoise, Snr) {
  const std::string tmpDir = getTmpPath("AdditiveNoise");
  dirCreateRecursive(tmpDir);
  const std::string listFilePath = pathsConcat(tmpDir, "noise.lst");
  const std::string noiseFilePath = pathsConcat(tmpDir, "noise.flac");

  const float signalAmplitude = -1.0;
  const int signalLen = 10;
  std::vector<float> signal(signalLen, signalAmplitude);
  const float noiseAmplitude = 1.0;
  const int noiseLen = 10;
  std::vector<float> noise(noiseLen, noiseAmplitude);

  saveSound(
      noiseFilePath,
      noise,
      sampleRate,
      1,
      fl::pkg::speech::SoundFormat::FLAC,
      fl::pkg::speech::SoundSubFormat::PCM_16);

  // Create test list file
  {
    std::ofstream listFile(listFilePath);
    listFile << noiseFilePath;
  }

  float threshold = 0.02; // allow 2% difference from expected value

  for (float snr = 1; snr < 30; ++snr) {
    AdditiveNoise::Config conf;
    conf.proba_ = 1.0;
    conf.ratio_ = 1.0;
    conf.minSnr_ = snr;
    conf.maxSnr_ = snr;
    conf.nClipsMin_ = 1;
    conf.nClipsMax_ = 1;
    conf.listFilePath_ = listFilePath;

    AdditiveNoise sfx(conf);
    auto augmented = signal;
    sfx.apply(augmented);

    std::vector<float> extractNoise(augmented.size());
    for (int i = 0; i < extractNoise.size(); ++i) {
      extractNoise[i] = (augmented[i] - signal[i]);
    }

    ASSERT_LE(
        signalToNoiseRatio(signal, extractNoise),
        (conf.maxSnr_ * (1 + threshold)));
    ASSERT_GE(
        signalToNoiseRatio(signal, extractNoise),
        (conf.minSnr_ * (1 - threshold)));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
