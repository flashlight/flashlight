/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/app/asr/augmentation/GaussianNoise.h"
#include "flashlight/app/asr/augmentation/SoundEffectUtil.h"

using namespace ::fl::app::asr::sfx;

const int numSamples = 10000;

TEST(GaussianNoise, SnrCheck) {
    RandomNumberGenerator rng(0);
    std::vector<float> signal(numSamples);
    for (auto& i : signal) {
        i = rng.random() ;
    }

    GaussianNoise::Config cfg;
    cfg.minSnr_ = 9;
    cfg.maxSnr_ = 11;
    GaussianNoise sfx(cfg);
    auto originalSignal = signal;
    sfx.apply(signal);
    ASSERT_EQ(signal.size(), originalSignal.size());
    std::vector<float> noise(signal.size());
    for (int i = 0 ;i < noise.size(); ++i) {
      noise[i] = signal[i] - originalSignal[i];
    }
    ASSERT_LE(signalToNoiseRatio(originalSignal, noise), cfg.maxSnr_);
    ASSERT_GE(signalToNoiseRatio(originalSignal, noise), cfg.minSnr_);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
