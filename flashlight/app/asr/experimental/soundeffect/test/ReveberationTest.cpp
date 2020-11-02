/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/experimental/soundeffect/Reverberation.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <sstream>
#include <string>

#include <arrayfire.h>
#include <flashlight/fl/flashlight.h>
#include <glog/logging.h>

#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/app/asr/experimental/soundeffect/Sound.h"
#include "flashlight/app/asr/experimental/soundeffect/SoundLoader.h"
#include "flashlight/app/asr/experimental/soundeffect/test/SoundTestUtil.h"
#include "flashlight/lib/common/System.h"

using namespace ::fl::app::asr::sfx;
using ::fl::lib::pathsConcat;

namespace {
std::string loadPath = "";
}

TEST(ReverbEcho, IterateParams) {
  auto soundLoader = std::make_shared<SoundLoaderRandomWithReplacement>(
      pathsConcat(loadPath, "librispeech/train-clean-10.lst"));

  ReverbEcho::Config conf;
  conf.lengthMilliseconds_ = 1000;
  conf.sampleRate_ = 16000;

  for (int numEchos = 1; numEchos <= 64; numEchos *= 4) {
    conf.numEchosMin_ = numEchos;
    conf.numEchosMax_ = numEchos;
    // Choosing very short distance such that we can print and visualize the
    // kernel.
    for (float distance = 0.1; distance <= 10; distance *= 10) {
      conf.distanceToWallInMetersMin_ = distance;
      conf.distanceToWallInMetersMax_ = distance;

      for (float absorptionCoefficient = 0.01; absorptionCoefficient <= 0.4;
           absorptionCoefficient *= 2) {
        conf.absorptionCoefficientMin_ = absorptionCoefficient;
        conf.absorptionCoefficientMax_ = absorptionCoefficient;

        ReverbEcho reverbEcho(conf);
        Sound augmented = reverbEcho.apply(soundLoader->loadRandom());
        std::stringstream filename;
        filename << "echo-" << numEchos << "-dist-" << distance << "-absrb-"
                 << absorptionCoefficient << ".flac";
        augmented.listFileEntry_.audioFilePath_ =
            pathsConcat(pathsConcat(loadPath, "ReverbEcho"), filename.str());
        augmented.writeToFile();
        FL_LOG(fl::INFO) << "Saved augmented file="
                         << augmented.listFileEntry_.audioFilePath_;
      }
    }
  }
}

TEST(ReverbDataset, ApplyDataset) {
  auto soundLoader = std::make_shared<SoundLoaderRandomWithReplacement>(
      pathsConcat(loadPath, "librispeech/train-clean-10.lst"));
  ReverbDataset::Config conf;
  conf.listFilePath_ = pathsConcat(loadPath, "BUT/reverb.lst");
  conf.randomRirWithReplacement_ = false;
  ReverbDataset reverbDataset(conf);
  Sound signal = soundLoader->loadRandom();

  for (int i = 0; i <= 10; ++i) {
    Sound augmented = reverbDataset.apply(signal.getCopy());
    std::stringstream filename;
    filename << "reverb-" << i << ".flac";
    augmented.listFileEntry_.audioFilePath_ =
        pathsConcat(pathsConcat(loadPath, "ReverbDataset"), filename.str());
    augmented.writeToFile();
    FL_LOG(fl::INFO) << "Saved augmented file="
                     << augmented.listFileEntry_.audioFilePath_;
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
