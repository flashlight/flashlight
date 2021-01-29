/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <math.h>

#include "flashlight/app/asr/augmentation/Reverberation.h"
#include "flashlight/app/asr/augmentation/SoundEffectUtil.h"
#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/fl/common/Init.h"
#include "flashlight/lib/common/System.h"

using namespace ::fl::app::asr::sfx;
using ::fl::lib::dirCreateRecursive;
using ::fl::lib::getTmpPath;
using ::fl::lib::pathsConcat;
using ::testing::AllOf;
using ::testing::Each;
using ::testing::Gt;
using ::testing::Lt;
using ::testing::Pointwise;

namespace {
// Arbitrary audioable signal values.
const int numSamples = 200;
const size_t freq = 1000;
const size_t sampleRate = 16000;
const float amplitude = 1.0;

std::vector<float>
genSinWave(size_t numSamples, size_t freq, size_t sampleRate, float amplitude) {
  std::vector<float> output(numSamples, 0);
  const float waveLenSamples =
      static_cast<float>(sampleRate) / static_cast<float>(freq);
  const float ratio = (2 * M_PI) / waveLenSamples;

  for (size_t i = 0; i < numSamples; ++i) {
    output.at(i) = amplitude * std::sin(static_cast<float>(i) * ratio);
  }
  return output;
}

} // namespace

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
  // Keep inital echo amplitude same as orig.
  conf.initialMin_ = 1;
  conf.initialMax_ = 1;

  std::vector<float> signal =
      genSinWave(numSamples, freq, sampleRate, amplitude);

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

/**
 * Tests that reverberation using randomly generated RIR yields the expected
 * noise. We generate a “signal” with value 1 at the zero location.
 * Reverberation over such a simple signal yields a vector containing the
 * The original signal plus the RIR concatenated with a vector of the same
 * length as the RIR. This whole output is scaled by a constant to keep the
 * energy intensity of the input.
 * The test verifies that:
 * - output length is the input length + the RIR length.
 * - the extracted noise, which is the ouput minus the input, is the same as the
 *   RIR multiplied by a constant.
 */
TEST(ReverbDataset, ImpulseReverb) {
  const std::string tmpDir = getTmpPath("ReverbDataset");
  dirCreateRecursive(tmpDir);
  const std::string listFilePath = pathsConcat(tmpDir, "rir.lst");
  const std::string rirFilePath = pathsConcat(tmpDir, "rir.flac");

  // Signal with value 1 at location zero makes the calculation of result very
  // easy. The added noise is simply the RIR itself.
  std::vector<float> signal(numSamples, 0);
  signal[0] = 1;

  // Generate a random RIR with exponential decay.
  const int rirLen = numSamples / 2;
  const float firstDelay = 0.0001;
  const float rt60 = (float)rirLen / (float)sampleRate;
  RandomNumberGenerator rng;
  std::vector<float> rir(rirLen, 0);
  float frac = 1;
  for (int i = 0; i < rir.size(); ++i) {
    float jitter = 1 + rng.uniform(-0.1, 0.1);
    const float attenuation = std::pow(10, -3 * jitter * firstDelay / rt60);
    frac *= attenuation;
    rir[i] = frac;
  }

  // Create a test list file pointing to the RIR as flac file.
  saveSound(
      rirFilePath,
      rir,
      sampleRate,
      1,
      fl::app::asr::SoundFormat::FLAC,
      fl::app::asr::SoundSubFormat::PCM_16);
  {
    std::ofstream listFile(listFilePath);
    listFile << rirFilePath;
  }

  ReverbDataset::Config conf{.proba_ = 1.0, .listFilePath_ = listFilePath};
  ReverbDataset sfx(conf);
  auto augmented = signal;
  sfx.apply(augmented);

  EXPECT_EQ(augmented.size(), signal.size() + rir.size() - 1);
  EXPECT_THAT(augmented, Each(AllOf(Gt(-1.0), Lt(1.0))));

  std::vector<float> extractNoise(rir.size());
  for (int i = 0; i < extractNoise.size(); ++i) {
    extractNoise[i] = (augmented[i] - signal[i]);
  }

  // To reduce test flakiness, we trim the edges of the extracted noise and the
  // rir such that we compare the center part.
  const size_t trimSize = 10;
  std::vector<float> extractNoiseCenter(
      extractNoise.begin() + trimSize, extractNoise.end() - trimSize);
  std::vector<float> rirCenter(rir.begin() + trimSize, rir.end() - trimSize);

  // The reverb output is scaled linearly by a constant ratio. If we multiply
  // the extract noise by that constant we should get the original rir back.
  // That constant is the ratio = rirCenter[0] / extractNoiseCenter[i] for any
  // i.
  float ratio = rirCenter[0] / extractNoiseCenter[0];
  // Scale the extracted noise by the constant ratio,
  std::transform(
      extractNoiseCenter.begin(),
      extractNoiseCenter.end(),
      extractNoiseCenter.begin(),
      [ratio](float f) -> float { return f * ratio; });

  // Expect the extracted noise to match the RIR.
  EXPECT_THAT(
      extractNoiseCenter, Pointwise(FloatNearPointwise(10e-3), rirCenter));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
