/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/app/asr/augmentation/ReverbAndNoise.h"
#include "flashlight/app/asr/augmentation/SoundEffectUtil.h"
#include "flashlight/fl/common/Init.h"

using namespace ::fl::app::asr::sfx;
using ::testing::Pointwise;

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
TEST(ReverbAndNoise, AdditiveNoiseSnr) {
  const float threshold = 0.2; // allow 20% difference from expected value

  const float signalAmplitude = -1.0;
  const int signalLen = 10000;
  auto signal = genTestSinWave(signalLen, /*freq=*/200, signalAmplitude);

  createTestListFile("ReverbAndNoise", "signal", {signal});

  const float noiseAmplitude = 1.0;
  const int noiseLen = 5000;
  auto noise = genTestSinWave(noiseLen, /*freq=*/250, noiseAmplitude);

  const auto noiseListFilePath =
      createTestListFile("ReverbAndNoise", "noise", {noise});

  const auto rir = createTestImpulseResponse(noiseLen);
  const auto rirListFilePath =
      createTestListFile("ReverbAndNoise", "rir", {rir});

  std::vector<std::vector<float>> extractNoises;
  std::vector<std::vector<float>> augs;
  for (float snr = 5; snr <= 35; snr += 10) {
    ReverbAndNoise::Config conf{.proba_ = 1.0,
                                .rirListFilePath_ = rirListFilePath,
                                .volume_ = 0,
                                .sampleRate_ = 16000,
                                .minSnr_ = snr,
                                .maxSnr_ = snr,
                                .nClipsMin_ = 1,
                                .nClipsMax_ = 1,
                                .noiseListFilePath_ = noiseListFilePath};

    ReverbAndNoise sfx(conf);
    auto augmented = signal;
    sfx.apply(augmented);
    augs.push_back(augmented);

    std::vector<float> extractNoise(augmented.size());
    for (int i = 0; i < extractNoise.size(); ++i) {
      extractNoise[i] = (augmented[i] - signal[i]);
    }
    extractNoises.push_back(extractNoise);

    LOG(INFO) << "snr=" << snr << " signalToNoiseRatio(signal, extractNoise)="
              << signalToNoiseRatio(signal, extractNoise);
    EXPECT_LE(
        signalToNoiseRatio(signal, extractNoise),
        (conf.maxSnr_ * (1 + threshold)));
    EXPECT_GE(
        signalToNoiseRatio(signal, extractNoise),
        (conf.minSnr_ * (1 - threshold)));
  }
  createTestListFile("ReverbAndNoise", "augmented", augs);
  createTestListFile("ReverbAndNoise", "extractNoises", extractNoises);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
