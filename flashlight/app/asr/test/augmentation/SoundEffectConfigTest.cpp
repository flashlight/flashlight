/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/app/asr/augmentation/SoundEffect.h"
#include "flashlight/app/asr/augmentation/SoundEffectConfig.h"
#include "flashlight/lib/common/System.h"

using namespace ::fl::app::asr::sfx;

/**
 * Test creation of arbitrary sound effect chain configuration, writing of that
 * configuration to a json file, and reading the json file to create the
 * configured sound effect chain.
 */
TEST(SoundEffectConfigFile, ReadWriteJson) {
  const std::string configFile = fl::lib::getTmpPath("sfxConfig.json");
  // This log line alllows the user to ispect the config file or copy/paste
  // configuration.
  std::cout << "configFile=" << configFile << std::endl;

  std::vector<SoundEffectConfig> sfxConf1(6);

  sfxConf1[0].type_ = kAdditiveNoise;
  sfxConf1[0].additiveNoiseConfig_.ratio_ = 0.8;
  sfxConf1[0].additiveNoiseConfig_.minSnr_ = 0;
  sfxConf1[0].additiveNoiseConfig_.maxSnr_ = 30;
  sfxConf1[0].additiveNoiseConfig_.nClipsMin_ = 0;
  sfxConf1[0].additiveNoiseConfig_.nClipsMax_ = 4;
  sfxConf1[0].additiveNoiseConfig_.listFilePath_ = "/dataset/noise.lst";
  sfxConf1[0].additiveNoiseConfig_.dsetRndPolicy_ =
      stringToRandomPolicy("with_replacment");
  sfxConf1[0].additiveNoiseConfig_.randomSeed_ = 1111;

  sfxConf1[1].type_ = kAmplify;
  sfxConf1[1].amplifyConfig_.ratioMin_ = 1;
  sfxConf1[1].amplifyConfig_.ratioMax_ = 10;
  sfxConf1[1].amplifyConfig_.randomSeed_ = 123;

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
  sfxConf1[3].reverbEchoConfig_.sampleRate_ = 1600;
  sfxConf1[3].reverbEchoConfig_.randomSeed_ = 42;

  sfxConf1[4].type_ = kNormalize;
  sfxConf1[4].normalizeOnlyIfTooHigh_ = false;

  sfxConf1[5].type_ = kTimeStretch;
  sfxConf1[5].timeStretchConfig_.proba_ = 1.0;
  sfxConf1[5].timeStretchConfig_.factorMin_ = 0.8;
  sfxConf1[5].timeStretchConfig_.factorMax_ = 1.5;
  sfxConf1[5].timeStretchConfig_.window_ = 20.0;
  sfxConf1[5].timeStretchConfig_.shift_ = 0.8;
  sfxConf1[5].timeStretchConfig_.fading_ = 0.25;
  sfxConf1[3].reverbEchoConfig_.sampleRate_ = 1600;
  sfxConf1[3].reverbEchoConfig_.randomSeed_ = 77;

  writeSoundEffectConfigFile(configFile, sfxConf1);
  const std::vector<SoundEffectConfig> sfxConf2 =
      readSoundEffectConfigFile(configFile);
  EXPECT_EQ(sfxConf1.size(), sfxConf2.size());

  std::shared_ptr<SoundEffect> sfx = createSoundEffect(sfxConf2);
  EXPECT_NE(sfx.get(), nullptr);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
