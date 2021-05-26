/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/pkg/speech/augmentation/SoundEffect.h"
#include "flashlight/fl/common/Init.h"
#include "flashlight/pkg/speech/augmentation/SoundEffectUtil.h"

using namespace ::fl::pkg::speech::sfx;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::Ge;
using ::testing::Le;

// Arbitrary audioable sound values.
const int numSamples = 10000;
const size_t freq = 1000;
const size_t sampleRate = 16000;

/**
 * Test that after clamping the amplitude the resulting amplitude is in [-1..1]
 * range. This test first generate a sine-wave with amplitude of 2.0 then clamp
 * it and verifies the result.
 */
TEST(SoundEffect, ClampAmplitude) {
  ClampAmplitude sfx;
  const float amplitude = 2.0;
  std::vector<float> signal =
      genTestSinWave(numSamples, freq, sampleRate, amplitude);
  sfx.apply(signal);
  EXPECT_THAT(signal, Each(AllOf(Ge(-1.0), Le(1.0))));
}

/**
 * Test that after normalizing the amplitude the resulting amplitude is in
 * [-1..1] range. This test first generate a sine-wave with amplitude of 2.0
 * then normalizes it and verifies the result.
 */
TEST(SoundEffect, NormalizeTooHigh) {
  Normalize sfx(/*onlyIfTooHigh=*/true);
  const float amplitude = 2.0;
  std::vector<float> signal =
      genTestSinWave(numSamples, freq, sampleRate, amplitude);
  sfx.apply(signal);
  EXPECT_THAT(signal, Each(AllOf(Ge(-1.0), Le(1.0))));
}

/**
 * Test that normalize configured with onlyIfTooHigh=true, leaves the signal
 * unchanged when the input is in valid range
 */
TEST(SoundEffect, NoNormalizeTooLow) {
  Normalize sfx(/*onlyIfTooHigh=*/true);
  const float amplitude = 0.5;
  std::vector<float> signal =
      genTestSinWave(numSamples, freq, sampleRate, amplitude);
  std::vector<float> signalCopy = signal;
  sfx.apply(signalCopy);
  EXPECT_EQ(signal, signalCopy);
}

/**
 * Test that normalize configured with onlyIfTooHigh=false, increases max
 * amplitude of sine-wave in range [-0.5, 0.5] to range [-1,1]
 */
TEST(SoundEffect, NormalizeTooLow) {
  Normalize sfx(/*onlyIfTooHigh=*/false);
  const float amplitude = 0.5;
  std::vector<float> signal =
      genTestSinWave(numSamples, freq, sampleRate, amplitude);
  sfx.apply(signal);
  EXPECT_THAT(signal, Each(AllOf(Ge(-1.0f), Le(1.0f))));
  EXPECT_THAT(signal, testing::Contains(-1.0f));
  EXPECT_THAT(signal, testing::Contains(1.0f));
}

/**
 * We expect that after amplification of sin wave in range [-1..1] the result is
 * in range of [-amp.ratioMax_.. amp.ratioMax_]. Also, after  running a number
 * of iterations, each with different random amplification with range, that we
 * see minimum amplification that is at least twice amp.ratioMin_ and that we
 * see maximum amplification that is at least half amp.ratioMax_
 */
TEST(SoundEffect, Amplify) {
  const float amplitude = 1.0;
  Amplify::Config conf;
  conf.ratioMin_ = amplitude / 10;
  conf.ratioMax_ = amplitude * 10;
  Amplify sfx(conf);

  std::vector<float> sound =
      genTestSinWave(numSamples, freq, sampleRate, amplitude);

  // get min/max amplification after 100 apply, each chooses a random value with
  // in range.
  float minMaxAbsAmp = conf.ratioMax_;
  float maxMaxAbsAmp = 0;
  for (int i = 0; i < 100; ++i) {
    std::vector<float> soundCopy = sound;
    sfx.apply(soundCopy);
    // Ensure that current augmentation amplitude is within expected range.
    EXPECT_THAT(
        soundCopy, Each(AllOf(Ge(-conf.ratioMax_), Le(conf.ratioMax_))));

    for (auto amp : soundCopy) {
      minMaxAbsAmp = std::min(std::fabs(amp), minMaxAbsAmp);
      maxMaxAbsAmp = std::max(std::fabs(amp), maxMaxAbsAmp);
    }
  }

  // Ensure that all random augmentations amplitudes are within expected
  // random range.  EXPECT_LT(minMaxAbsAmp, amp.ratioMin_ * 2);
  EXPECT_GT(maxMaxAbsAmp, conf.ratioMax_ / 2);
}

// Test that basic sound effect chain processes in the correct order.
// We generate signal, amplify above valid range, then clamp it to valid
// range (-1..1) and multiply by amplitude. We expect that the result is
// in the range of: -amplitude..amplitude.
TEST(SoundEffect, SfxChain) {
  const float amplitude = 2.0;
  Amplify::Config amp1;
  amp1.ratioMin_ = amplitude / 10;
  amp1.ratioMax_ = amplitude * 10;
  Amplify::Config amp2;
  amp2.ratioMin_ = amplitude;
  amp2.ratioMax_ = amplitude;

  auto sfxChain = std::make_shared<SoundEffectChain>();
  sfxChain->add(std::make_shared<Amplify>(amp1));
  sfxChain->add(std::make_shared<ClampAmplitude>());
  sfxChain->add(std::make_shared<Amplify>(amp2));

  std::vector<float> signal =
      genTestSinWave(numSamples, freq, sampleRate, amplitude);
  sfxChain->apply(signal);
  EXPECT_THAT(signal, Each(AllOf(Ge(-amplitude), Le(amplitude))));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
