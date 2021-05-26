/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>

#include "flashlight/pkg/speech/augmentation/SoundEffect.h"
#include "flashlight/pkg/speech/augmentation/SoundEffectConfig.h"
#include "flashlight/fl/common/Init.h"
#include "flashlight/lib/common/System.h"

using namespace ::fl::pkg::speech::sfx;

/**
 * Test creation of arbitrary sound effect chain configuration, writing of that
 * configuration to a json file, and reading the json file to create the
 * configured sound effect chain.
 */
TEST(SoundEffectConfigFile, ReadWriteJson) {
  const std::string configPath = fl::lib::getTmpPath("sfxConfig.json");
  // This log line alllows the user to inspect the config file or copy/paste
  // configuration.
  LOG(INFO) << "output config file= " << configPath;

  std::vector<SoundEffectConfig> sfxConf1(6);

  // Create mock noise list file.
  const std::string noiseListPath = fl::lib::getTmpPath("noise.lst");
  {
    std::ofstream noiseListFile(noiseListPath);
    noiseListFile << "/fake/path.flac";
  }
  sfxConf1[0].type_ = kAdditiveNoise;
  sfxConf1[0].additiveNoiseConfig_.ratio_ = 0.8;
  sfxConf1[0].additiveNoiseConfig_.minSnr_ = 0;
  sfxConf1[0].additiveNoiseConfig_.maxSnr_ = 30;
  sfxConf1[0].additiveNoiseConfig_.nClipsMin_ = 0;
  sfxConf1[0].additiveNoiseConfig_.nClipsMax_ = 4;
  sfxConf1[0].additiveNoiseConfig_.listFilePath_ = noiseListPath;

  sfxConf1[1].type_ = kAmplify;
  sfxConf1[1].amplifyConfig_.ratioMin_ = 1;
  sfxConf1[1].amplifyConfig_.ratioMax_ = 10;

  sfxConf1[2].type_ = kClampAmplitude;

  sfxConf1[3].type_ = kReverbEcho;
  sfxConf1[3].reverbEchoConfig_.proba_ = 0.5;
  sfxConf1[3].reverbEchoConfig_.initialMin_ = 0.1;
  sfxConf1[3].reverbEchoConfig_.initialMax_ = 0.3;
  sfxConf1[3].reverbEchoConfig_.rt60Min_ = 0.3;
  sfxConf1[3].reverbEchoConfig_.rt60Max_ = 1.3;
  sfxConf1[3].reverbEchoConfig_.firstDelayMin_ = 0.01;
  sfxConf1[3].reverbEchoConfig_.firstDelayMax_ = 0.03;
  sfxConf1[3].reverbEchoConfig_.repeat_ = 3;
  sfxConf1[3].reverbEchoConfig_.jitter_ = 0.2;
  sfxConf1[3].reverbEchoConfig_.sampleRate_ = 16000;

  sfxConf1[4].type_ = kNormalize;
  sfxConf1[4].normalizeOnlyIfTooHigh_ = false;

  sfxConf1[5].type_ = kTimeStretch;
  sfxConf1[5].timeStretchConfig_.proba_ = 1.0;
  sfxConf1[5].timeStretchConfig_.minFactor_ = 0.8;
  sfxConf1[5].timeStretchConfig_.maxFactor_ = 1.5;

  writeSoundEffectConfigFile(configPath, sfxConf1);
  const std::vector<SoundEffectConfig> sfxConf2 =
      readSoundEffectConfigFile(configPath);
  EXPECT_EQ(sfxConf1.size(), sfxConf2.size());

  std::shared_ptr<SoundEffect> sfx = createSoundEffect(sfxConf2);
  EXPECT_NE(sfx.get(), nullptr);
}

int main(int argc, char** argv) {
  fl::init();
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
