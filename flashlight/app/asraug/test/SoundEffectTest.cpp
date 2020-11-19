/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/app/asraug/SoundEffect.h"
#include "flashlight/app/asraug/test/SoundTestUtil.h"

using namespace fl::app::asr;
using namespace ::fl::app::asr::sfx;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::Le;
using ::testing::Lt;

// Arbitrary audioable sound values.
const int numSamples = 1000;
const size_t freq = 1000;
const size_t sampleRate = 16000;

TEST(SoundEffect, ClampAmplitude) {
  sfx::ClampAmplitude sfx;
  const float amplitude = 2.0;
  std::vector<float> signal =
      genSinWave(numSamples, freq, sampleRate, amplitude);
  sfx.apply(&signal);
  EXPECT_THAT(signal, Each(AllOf(Ge(-1.0), Le(1.0))));
}

TEST(SoundEffect, NormalizeTooHigh) {
  sfx::Normalize sfx(/*onlyIfTooHigh=*/true);
  const float amplitude = 2.0;
  std::vector<float> signal =
      genSinWave(numSamples, freq, sampleRate, amplitude);
  sfx.apply(&signal);
  EXPECT_THAT(signal, Each(AllOf(Ge(-1.0), Le(1.0))));
}

TEST(SoundEffect, NoNormalizeTooLow) {
  sfx::Normalize sfx(/*onlyIfTooHigh=*/true);
  const float amplitude = 0.5;
  std::vector<float> signal =
      genSinWave(numSamples, freq, sampleRate, amplitude);
  sfx.apply(&signal);
  EXPECT_THAT(signal, Each(AllOf(Ge(-amplitude), Le(amplitude))));
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
  sfx::Amplify::Config amp;
  amp.ratioMin_ = amplitude / 10;
  amp.ratioMax_ = amplitude * 10;
  sfx::Amplify sfx(amp);

  std::vector<float> sound =
      genSinWave(numSamples, freq, sampleRate, amplitude);

  // get min/max amplification after 100 apply, each chooses a random value with
  // in range.
  float minMaxAbsAmp = amp.ratioMax_;
  float maxMaxAbsAmp = 0;
  for (int i = 0; i < 100; ++i) {
    std::vector<float> soundCopy = sound;
    sfx.apply(&soundCopy);
    // Ensure that current augmentation amplitude is within expected range.
    EXPECT_THAT(soundCopy, Each(AllOf(Ge(-amp.ratioMax_), Le(amp.ratioMax_))));

    for (auto amp : soundCopy) {
      minMaxAbsAmp = std::min(std::fabs(amp), minMaxAbsAmp);
      maxMaxAbsAmp = std::max(std::fabs(amp), maxMaxAbsAmp);
    }
  }

  // Ensure that all random augmentations amplitudes are within expected
  // random range.  EXPECT_LT(minMaxAbsAmp, amp.ratioMin_ * 2);
  EXPECT_GT(maxMaxAbsAmp, amp.ratioMax_ / 2);
}

// Test that basic sound effect chain processes in the correct order.
// We generate singal, amplify above valid range, then clamp it to valid
// range (-1..1) and multiply by amplitude. We expect that the result is
// in the range of: -amplitude..amplitude.
TEST(SoundEffect, SfxChain) {
  const float amplitude = 2.0;
  sfx::Amplify::Config amp1;
  amp1.ratioMin_ = amplitude / 10;
  amp1.ratioMax_ = amplitude * 10;
  sfx::Amplify::Config amp2;
  amp2.ratioMin_ = amplitude;
  amp2.ratioMax_ = amplitude;

  auto sfxChain = std::make_shared<SoundEffectChain>();
  sfxChain->add(std::make_shared<sfx::Amplify>(amp1));
  sfxChain->add(std::make_shared<sfx::ClampAmplitude>());
  sfxChain->add(std::make_shared<sfx::Amplify>(amp2));

  std::vector<float> signal =
      genSinWave(numSamples, freq, sampleRate, amplitude);
  sfxChain->apply(&signal);
  EXPECT_THAT(signal, Each(AllOf(Ge(-amplitude), Le(amplitude))));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
