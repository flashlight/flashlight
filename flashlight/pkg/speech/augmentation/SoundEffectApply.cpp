/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/pkg/speech/augmentation/SoundEffect.h"
#include "flashlight/pkg/speech/augmentation/SoundEffectConfig.h"
#include "flashlight/pkg/speech/data/Sound.h"

DEFINE_string(input, "", "Sound file to augment.");
DEFINE_string(
    output,
    "augmented.flac",
    "Path to store result of augmenting the input file");
DEFINE_string(config, "", "Path to a sound effect json config file");

using namespace ::fl::pkg::speech::sfx;
using ::fl::pkg::speech::loadSound;
using ::fl::pkg::speech::loadSoundInfo;
using ::fl::pkg::speech::saveSound;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);

  gflags::SetUsageMessage(
      "Usage: \n " + exec +
      " --input=[path to input file] --output=[path to output file] " +
      "--config=[path to config file]");

  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  gflags::ParseCommandLineFlags(&argc, &argv, false);

  if (FLAGS_config.empty()) {
    LOG(FATAL) << "flag --config must point to sound effect config file";
  }
  if (FLAGS_input.empty()) {
    LOG(FATAL) << "flag --input must point to input file";
  }

  auto sound = loadSound<float>(FLAGS_input);
  auto info = loadSoundInfo(FLAGS_input);

  std::shared_ptr<SoundEffect> sfx =
      createSoundEffect(readSoundEffectConfigFile(FLAGS_config));
  sfx->apply(sound);

  saveSound(
      FLAGS_output,
      sound,
      info.samplerate,
      info.channels,
      fl::pkg::speech::SoundFormat::FLAC,
      fl::pkg::speech::SoundSubFormat::PCM_16);

  LOG(INFO) << "Saving augmented file to=" << FLAGS_output;

  return 0;
}
