/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/pkg/speech/augmentation/GaussianNoise.h"
#include "flashlight/pkg/speech/augmentation/SoundEffectUtil.h"
#include "flashlight/fl/common/Init.h"

using namespace ::fl::pkg::speech::sfx;

const int numSamples = 10000;

TEST(GaussianNoise, SnrCheck) {
  int numTrys = 10;
  float tolerance = 1e-1;
  // Use `r` as seed so that we test different input samples at different SNRs
  for (int r = 0; r < numTrys; ++r) {
    RandomNumberGenerator rng(r);
    std::vector<float> signal(numSamples);
    for (auto& i : signal) {
        i = rng.random() ;
    }

    GaussianNoise::Config cfg;
    cfg.minSnr_ = 8;
    cfg.maxSnr_ = 12;
    GaussianNoise sfx(cfg, r);
    auto originalSignal = signal;
    sfx.apply(signal);
    ASSERT_EQ(signal.size(), originalSignal.size());
    std::vector<float> noise(signal.size());
    for (int i = 0 ;i < noise.size(); ++i) {
      noise[i] = signal[i] - originalSignal[i];
    }
    ASSERT_LE(signalToNoiseRatio(originalSignal, noise), cfg.maxSnr_ + tolerance);
    ASSERT_GE(signalToNoiseRatio(originalSignal, noise), cfg.minSnr_ - tolerance);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
