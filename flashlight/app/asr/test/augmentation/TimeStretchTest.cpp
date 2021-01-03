/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/app/asr/augmentation/TimeStretch.h"
#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/lib/common/System.h"

using namespace ::fl::app::asr::sfx;

namespace {
// Arbitrary audioable signal values.
const int numSamples = 20000;
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

TEST(TimeStretch, Basic) {
  std::vector<float> signal =
      genSinWave(numSamples, freq, sampleRate, amplitude);

  for (int leaveUnchanged = 0; leaveUnchanged < 2; ++leaveUnchanged) {
    for (float factor = 0.7; factor <= 1.3; factor += 0.1f) {
      TimeStretch::Config conf;
      conf.factorMin_ = factor;
      conf.factorMax_ = factor;
      conf.leaveLengthUnchanged_ = (leaveUnchanged != 0);

      TimeStretch sfx(conf);
      auto augmented = signal;
      sfx.apply(augmented);

      if (leaveUnchanged) {
        EXPECT_EQ(signal.size(), augmented.size());
      } else {
        EXPECT_FLOAT_EQ(
            static_cast<float>(augmented.size()) / signal.size(), factor);
      }
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
