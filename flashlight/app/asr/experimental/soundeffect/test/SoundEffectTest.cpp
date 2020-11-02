/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/app/asr/experimental/soundeffect/AdditiveNoise.h"
#include "flashlight/app/asr/experimental/soundeffect/Sound.h"
#include "flashlight/app/asr/experimental/soundeffect/test/SoundTestUtil.h"

using namespace fl::app::asr;
using namespace ::fl::app::asr::sfx;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::Le;
using ::testing::Lt;
using ::testing::Not;
using ::testing::Pointwise;

namespace {
std::string loadPath = "";
}

// Arbitrary audioable sound values.
const int numSamples = 1000;
const size_t freq = 1000;
const size_t sampleRate = 16000;
const float amplitude = 0.5;

class SoundGetBackendTestFixture : public ::testing::TestWithParam<Backend> {
 protected:
  // Create identical sound data for orig_ and origSave_
  void SetUp() override {
    orig_ = genSinWave(numSamples, freq, sampleRate, amplitude);
    origSave_ = genSinWave(numSamples, freq, sampleRate, amplitude);
  }

  // Move data back to backend if not there.
  void restoreBackend(Backend backend) {
    if (backend == Backend::CPU) {
      orig_.getCpuData();
      origSave_.getCpuData();
    } else {
      orig_.getGpuData();
      origSave_.getGpuData();
    }
  }

  Sound orig_;
  Sound origSave_;
};

// Test that getCopy() creates a separate copy.
TEST_P(SoundGetBackendTestFixture, GetCopy) {
  const Backend backend = GetParam();

  restoreBackend(backend);

  // Expect orig == origSave
  EXPECT_THAT(*orig_.getCpuData(), Pointwise(Eq(), *origSave_.getCpuData()));

  restoreBackend(backend);

  Sound copy = orig_.getCopy();
  // Expect copy == orig
  EXPECT_THAT(*orig_.getCpuData(), Pointwise(Eq(), *copy.getCpuData()));

  restoreBackend(backend);

  copy /= 2.0;

  // Expect orig == origSave
  EXPECT_THAT(*orig_.getCpuData(), Pointwise(Eq(), *origSave_.getCpuData()));

  restoreBackend(backend);

  // Expect copy != orig
  EXPECT_THAT(*orig_.getCpuData(), Not(Pointwise(Eq(), *copy.getCpuData())));
}

// Test on CPU and GPU backends.
INSTANTIATE_TEST_SUITE_P(
    SoundGetBackendTest,
    SoundGetBackendTestFixture,
    ::testing::Values(Backend::GPU, Backend::CPU));

TEST(SoundEffect, ClampAmplitude) {
  const float amplitude = 2.0;
  Sound signal = genSinWave(numSamples, freq, sampleRate, amplitude);
  auto sfx = std::make_shared<sfx::ClampAmplitude>();
  Sound augmented = sfx->apply(signal);
  EXPECT_THAT(*augmented.getCpuData(), Each(AllOf(Ge(-1.0), Le(1.0))));
}

TEST(SoundEffect, NormalizeTooHigh) {
  const float amplitude = 2.0;
  Sound signal = genSinWave(numSamples, freq, sampleRate, amplitude);
  auto sfx = std::make_shared<sfx::Normalize>();
  Sound augmented = sfx->apply(signal);
  EXPECT_THAT(*augmented.getCpuData(), Each(AllOf(Ge(-1.0), Le(1.0))));
}

TEST(SoundEffect, NoNormalizeTooLow) {
  const float amplitude = 0.5;
  Sound signal = genSinWave(numSamples, freq, sampleRate, amplitude);
  auto sfx = std::make_shared<sfx::Normalize>();
  Sound augmented = sfx->apply(signal);
  EXPECT_THAT(
      *augmented.getCpuData(), Each(AllOf(Ge(-amplitude), Le(amplitude))));
}

TEST(SoundEffect, SetEnableCountdown) {
  const float amplitude = 2.0;
  const int enableCountdown = 10;
  auto sfx = std::make_shared<sfx::Normalize>();
  sfx->setEnableCountdown(enableCountdown);
  for (int i = 0; i < enableCountdown * 2; ++i) {
    Sound signal = genSinWave(numSamples, freq, sampleRate, amplitude);
    Sound augmented = sfx->apply(signal);

    if (i < enableCountdown) {
      EXPECT_THAT(
          *augmented.getCpuData(),
          AllOf(Contains(Lt(-1.0)), Contains(Gt(1.0))));
    } else {
      EXPECT_THAT(*augmented.getCpuData(), Each(AllOf(Ge(-1.0), Le(1.0))));
    }
  }
}

// Test that basic sound effect chain processes in the correct order.
// We generate singal, normalizes it (to -1..1 value range) and multiply by
// amplitude. We expect that the result is in the range of:
// -amplitude..amplitude.
TEST(SoundEffect, SfxChain) {
  const float amplitude = 2.0;
  const int enableCountdown = 10;

  auto sfx = std::make_shared<sfx::Normalize>();

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

  sfx->setEnableCountdown(enableCountdown);
  for (int i = 0; i < enableCountdown * 2; ++i) {
    Sound signal = genSinWave(numSamples, freq, sampleRate, amplitude);
    Sound augmented = sfxChain->apply(signal);

    if (i < enableCountdown) {
      EXPECT_THAT(
          *augmented.getCpuData(),
          AllOf(Contains(Lt(-1.0)), Contains(Gt(1.0))));
    } else {
      EXPECT_THAT(
          *augmented.getCpuData(), Each(AllOf(Ge(-amplitude), Le(amplitude))));
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);

// Resolve directory for data
#ifdef DATA_TEST_DATADIR
  loadPath = DATA_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
