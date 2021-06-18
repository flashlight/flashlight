/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <thread>

#include "flashlight/lib/audio/feature/Dither.h"
#include "flashlight/lib/test/audio/feature/TestUtils.h"

using fl::lib::audio::Dither;

TEST(DitherTest, basicTest) {
  int N = 1000;

  for (int bch = 1; bch <= 8; bch *= 2) {
    Dither ditherpos(0.01);
    auto input = randVec<float>(N * bch);
    auto output = ditherpos.apply(input);
    // Dithering should change input slightly.
    ASSERT_FALSE(compareVec<float>(output, input, 1E-5));

    Dither ditherpos2(0.01);
    auto output2 = ditherpos2.apply(input);
    // Dither constant > 0 should give same result in multiple runs
    ASSERT_TRUE(compareVec<float>(output, output2, 1E-5));
  }

  for (int bch = 1; bch <= 8; bch *= 2) {
    Dither ditherneg(-0.01);
    auto input = randVec<float>(N * bch);
    auto output = ditherneg.apply(input);
    // Dithering should change input slightly.
    ASSERT_FALSE(compareVec<float>(output, input, 1E-6));

    // time(NULL) resolution is in seconds
    std::chrono::seconds dura(2);
    std::this_thread::sleep_for(dura);

    Dither ditherneg2(-0.01);
    auto output2 = ditherneg2.apply(input);
    // Dithering should change input slightly.
    ASSERT_FALSE(compareVec<float>(output, input, 1E-6));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
