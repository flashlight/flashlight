/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/experimental/soundeffect/AdditiveNoise.h"

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>

#include <arrayfire.h>
#include <flashlight/fl/flashlight.h>

#include "flashlight/app/asr/experimental/soundeffect/ListFile.h"
#include "flashlight/app/asr/experimental/soundeffect/Sound.h"
#include "flashlight/app/asr/experimental/soundeffect/SoundLoader.h"
#include "flashlight/app/asr/experimental/soundeffect/test/SoundTestUtil.h"
#include "flashlight/lib/common/System.h"

using namespace ::fl::app::asr::sfx;
using ::fl::lib::pathsConcat;

namespace {
std::string loadPath = "/tmp";
}

TEST(AdditiveNoise, AugmentSpeech) {
  auto soundLoader = std::make_shared<SoundLoaderRandomWithReplacement>(
      pathsConcat(loadPath, "librispeech/train-clean-10.lst"));

  AdditiveNoise::Config conf;
  conf.listFilePath_ =
      pathsConcat(loadPath, "audioset/balanced-16kHz-mono.lst");

  conf.randomNoiseWithReplacement_ = true;

  for (float maxTimeRatio = 0.5; maxTimeRatio <= 1; maxTimeRatio += 0.5) {
    conf.maxTimeRatio_ = maxTimeRatio;
    for (int nClipsPerUtterance = 1; nClipsPerUtterance < 4;
         ++nClipsPerUtterance) {
      conf.nClipsPerUtteranceMin_ = nClipsPerUtterance;
      conf.nClipsPerUtteranceMax_ = nClipsPerUtterance;
      for (int snr = -20; snr <= 30; snr += 5) {
        conf.minSnr_ = snr;
        conf.maxSnr_ = snr;
        AdditiveNoise additiveNoise(conf);

        Sound augmented = additiveNoise.apply(soundLoader->loadRandom());
        std::stringstream filename;
        filename << "clips-" << nClipsPerUtterance << "-snr-" << snr
                 << "-maxTimeRatio-" << maxTimeRatio << ".flac";
        augmented.listFileEntry_.audioFilePath_ =
            pathsConcat(pathsConcat(loadPath, "AdditiveNoise"), filename.str());
        augmented.writeToFile();
        FL_LOG(fl::INFO) << "Saved augmented file="
                         << augmented.listFileEntry_.audioFilePath_;
      }
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);

// Resolve directory for data
#ifdef DATA_TEST_DATADIR
  loadPath = DATA_TEST_DATADIR;
#else
  loadPath = "/tmp"
#endif

  return RUN_ALL_TESTS();
}
