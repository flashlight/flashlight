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
#include "flashlight/app/asr/experimental/soundeffect/SoundEffectConfigFile.h"
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

TEST(SoundEffectConfigFile, AdditiveNoise) {
  const std::string debugDir = testFilename("debug");
  const std::string configFile = testFilename("additiveNoise.json");

  std::vector<fl::app::asr::sfx::SoundEffectConfig> sfxConf1(5);
  sfxConf1[0].type_ = kAdditiveNoise;
  sfxConf1[0].noiseConfig_.randomSeed_ = 1234;
  sfxConf1[0].noiseConfig_.minSnr_ = 0;
  sfxConf1[0].noiseConfig_.maxSnr_ = 10;
  sfxConf1[0].noiseConfig_.nClipsPerUtteranceMin_ = 0;
  sfxConf1[0].noiseConfig_.nClipsPerUtteranceMax_ = 1;
  sfxConf1[0].noiseConfig_.listFilePath_ =
      "/private/home/wesbz/experiments/noise-unbalanced-16kHz-mono-train.lst";

  sfxConf1[1].type_ = kReverbEcho;
  sfxConf1[1].reverbEchoConfig_.randomSeed_ = 1234567;
  sfxConf1[1].reverbEchoConfig_.absorptionCoefficientMin_ = 0.01;
  sfxConf1[1].reverbEchoConfig_.absorptionCoefficientMax_ = 0.1;
  sfxConf1[1].reverbEchoConfig_.distanceToWallInMetersMin_ = 1;
  sfxConf1[1].reverbEchoConfig_.distanceToWallInMetersMax_ = 10;
  sfxConf1[1].reverbEchoConfig_.numEchosMin_ = 0;
  sfxConf1[1].reverbEchoConfig_.numEchosMax_ = 10;
  sfxConf1[1].reverbEchoConfig_.jitter_ = 0.1;

  sfxConf1[2].type_ = kReverbDataset;
  sfxConf1[2].reverbDatasetConfig_.randomSeed_ = 12;
  sfxConf1[2].reverbDatasetConfig_.listFilePath_ =
      "/private/home/avidov/fbcode/deeplearning/projects/flashlight/flashlight/app/asr"
      "/experimental/soundeffect/test/data/BUT/reverb.lst";
  sfxConf1[2].reverbDatasetConfig_.randomRirWithReplacement_ = true;

  sfxConf1[3].type_ = kAmplify;
  sfxConf1[3].amplifyConfig_.randomSeed_ = 12;
  sfxConf1[3].amplifyConfig_.ratioMin_ = 0.1;
  sfxConf1[3].amplifyConfig_.ratioMax_ = 0.3;

  sfxConf1[4].type_ = kNormalize;

  writeSoundEffectConfigFile(configFile, sfxConf1);
  FL_LOG(fl::INFO) << "Saving configFile=" << configFile;
  const std::vector<fl::app::asr::sfx::SoundEffectConfig> sfxConf2 =
      readSoundEffectConfigFile(configFile);

  std::shared_ptr<SoundEffect> sfx = createSoundEffect(sfxConf2);
  FL_LOG(fl::INFO) << "sfx:\n" << sfx->prettyString();

  EXPECT_EQ(sfxConf1.size(), sfxConf2.size());
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
