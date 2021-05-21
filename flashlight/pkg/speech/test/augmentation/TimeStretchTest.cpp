/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/pkg/speech/augmentation/SoundEffectUtil.h"
#include "flashlight/pkg/speech/augmentation/TimeStretch.h"
#include "flashlight/fl/common/Init.h"

using namespace ::fl::app::asr::sfx;

// Arbitrary audioable signalSoxFmt values.
const int numSamples = 20000;
const size_t freq = 1000;
const size_t sampleRate = 16000;
const float amplitude = 0.5;

/**
 * Test that stretched signal is of expected size.
 * - Generate base signal
 * - stretch it with diffret factors.
 * - Test that length of stretched signal is the length of the base
 *      signal times the factor.
 */
TEST(TimeStretch, SinWave) {
  float tolerance = 0.05;

  const std::vector<float> signal =
      genTestSinWave(numSamples, freq, sampleRate, amplitude);

  for (float factor = 0.5; factor <= 2; factor += 0.1) {
    std::vector<float> augmented = signal;

    TimeStretch::Config conf = {
        .proba_ = 1.0, .minFactor_ = factor, .maxFactor_ = factor};
    TimeStretch sfx(conf);
    sfx.apply(augmented);

    const float stretchRatio = static_cast<float>(augmented.size()) /
        static_cast<float>(signal.size());

    EXPECT_GE(stretchRatio, factor * (1 - tolerance));
    EXPECT_LE(stretchRatio, factor * (1 + tolerance));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
