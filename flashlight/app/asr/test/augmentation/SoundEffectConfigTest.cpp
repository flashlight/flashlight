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
using ::fl::lib::pathsConcat;

/**
 * Test creation of arbitrary sound effect chain configuration, writing of that
 * configuration to a json file, and reading the json file to create the
 * configured sound effect chain.
 */
TEST(SoundEffectConfigFile, ReadWriteJson) {
  const std::string configFile =
      pathsConcat(std::tmpnam(nullptr), "sfxConfig.json");
  // This log line alllows the user to ispect the config file or copy/paste
  // configuration.
  std::cout << "configFile=" << configFile << std::endl;

  std::vector<SoundEffectConfig> sfxConf1(3);
  sfxConf1[0].type_ = kClampAmplitude;

  sfxConf1[1].type_ = kAmplify;
  sfxConf1[1].amplifyConfig_.randomSeed_ = 123;
  sfxConf1[1].amplifyConfig_.ratioMin_ = 1;
  sfxConf1[1].amplifyConfig_.ratioMax_ = 10;

  sfxConf1[2].type_ = kNormalize;
  sfxConf1[2].normalizeOnlyIfTooHigh_ = false;

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
