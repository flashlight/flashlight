/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/pkg/speech/augmentation/Reverberation.h"
#include "flashlight/pkg/speech/augmentation/SoundEffectUtil.h"
#include "flashlight/fl/common/Init.h"

using namespace ::fl::pkg::speech::sfx;
using testing::Pointwise;

// Arbitrary audioable signal values.
const int numSamples = 200;
const size_t freq = 1000;
const size_t sampleRate = 16000;
const float amplitude = 1.0;

MATCHER_P(FloatNearPointwise, tol, "Out of range") {
  return (
      std::get<0>(arg) > std::get<1>(arg) - tol &&
      std::get<0>(arg) < std::get<1>(arg) + tol);
}

/**
 * Test that reverberations augments the signal correctly over time and
 * amplitude dimensions. On time dimension we test that the signal is kept
 * unchanged before firstDelayMin_ but it does change after that period. On
 * amplitude dimension we test that the reverb noise matches the input signal.
 * For testing we need deterministic output from this random sound effect. For
 * that purpose we:
 * - set all random intervals to zero length.
 * - set a very long rt60 such that attenuation is nearly non-existent over the
 *   period of the input.
 * To keep calculation simple we also set the initial reflection ratio to 1. The
 * very long rt60 in combination with multiple repeats (repeat_ > 1) and high
 * initial reflection value, causes reflections to be of very high magnitude and
 * thus the reverb output amplitude can be much higher than the input. This
 * should not happen with realistic rt60 and initial values but makes test much
 * simpler.
 */
TEST(ReverbEcho, SinWaveReverb) {
  // Make the reverb start at the center of the sample vector.
  const size_t firstReverbIdx = numSamples / 2;
  const float firstDelay =
      static_cast<float>(firstReverbIdx) / static_cast<float>(sampleRate);

  ReverbEcho::Config conf;
  conf.proba_ = 1.0f; // revern every sample
  // Force delay to a specific period
  conf.firstDelayMin_ = firstDelay;
  conf.firstDelayMax_ = firstDelay;
  // No jitter so delay is deterministic
  conf.jitter_ = 0;
  // Make very long rt60 so attenuation over the period of the signal is nearly
  // zero.
  conf.rt60Min_ = firstDelay * 100;
  conf.rt60Min_ = firstDelay * 100;
  conf.repeat_ = 3;
  // Keep inital echo aplitude same as orig.
  conf.initialMin_ = 1;
  conf.initialMax_ = 1;

  std::vector<float> signal =
      genTestSinWave(numSamples, freq, sampleRate, amplitude);

  std::vector<float> input = signal;
  std::vector<float> inpuBeforeDelay(
      signal.begin(), signal.begin() + firstReverbIdx - 1);
  std::vector<float> inpuAfterDelay(
      signal.begin() + firstReverbIdx, signal.end());

  ReverbEcho sfx(conf);
  sfx.apply(signal);

  std::vector<float> outputBeforeDelay(
      signal.begin(), signal.begin() + firstReverbIdx - 1);
  std::vector<float> outputAfterDelay(
      signal.begin() + firstReverbIdx, signal.end());

  EXPECT_EQ(inpuBeforeDelay, outputBeforeDelay);
  EXPECT_NE(inpuAfterDelay, outputAfterDelay);

  // Extract the noise and compare with input that is the source of that noise.
  std::vector<float> noise(firstReverbIdx);
  for (int k = firstReverbIdx; k < signal.size(); ++k) {
    noise[k - firstReverbIdx] = signal[k] - input[k];
  }
  // Because we use very long rt60 and we use multiple repeasts, the reverb sum
  // can get to very high values. We normalize by mean of the abs diffs.
  float noiseSum = 0;
  float inputSum = 0;
  for (int j = firstReverbIdx; j < signal.size(); ++j) {
    noiseSum += std::abs(signal[j] - input[j]);
    inputSum += std::abs(input[j - firstReverbIdx]);
  }
  float norm = noiseSum / inputSum;
  std::transform(
      noise.begin(), noise.end(), noise.begin(), [norm](float x) -> float {
        return x / norm;
      });

  // To reduce test flakiness, we trim the edges of the noise and compare only
  // with the part in the input that is the source of this reverb noise.
  std::vector<float> noiseMain(noise.begin() + 10, noise.end() - 10);
  std::vector<float> noiseSrc(
      input.begin() + 9, input.begin() + firstReverbIdx - 11);

  EXPECT_EQ(noiseMain.size(), noiseSrc.size());
  EXPECT_THAT(noiseMain, Pointwise(FloatNearPointwise(0.1), noiseSrc));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
