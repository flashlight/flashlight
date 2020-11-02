/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <flashlight/fl/flashlight.h>
#include <gflags/gflags.h>

#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/experimental/soundeffect/ListFile.h"
#include "flashlight/app/asr/experimental/soundeffect/Sound.h"
#include "flashlight/app/asr/experimental/soundeffect/SoundEffect.h"
#include "flashlight/app/asr/experimental/soundeffect/SoundEffectConfigFile.h"
#include "flashlight/app/asr/experimental/soundeffect/SoundLoader.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/lib/common/System.h"

DECLARE_string(input_rootdir);
DECLARE_string(input_listfiles);
DECLARE_string(output_rootdir);
DECLARE_string(output_listfile);

DEFINE_string(
    input_rootdir,
    "",
    "directory that is used as prefix to input files that are relative path.");
DEFINE_string(
    input_listfiles,
    "",
    "commas seperated list of input list files. Content of all files is updated to the output file.");
DEFINE_string(
    output_rootdir,
    "",
    "directory that is used as prefix to output files that are relative path.");
DEFINE_string(
    output_listfile,
    "augmented.lst",
    "filename for create output list file.");

using namespace ::fl::app::asr::sfx;
using namespace ::fl::app::asr;
using ::fl::Logging;
using ::fl::lib::pathsConcat;
using ::fl::lib::pathSeperator;
using ::fl::LogLevel::ERROR;
using ::fl::LogLevel::INFO;

std::shared_ptr<SoundEffectChain> createConfiguredSoundEffectChain();

int main(int argc, char** argv) {
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage(
      "Usage: \n " + exec +
      " --input_rootdir=[input root dir] --input_listfiles=[comma seperated list files]" +
      " --output_rootdir=[output root dir] --output_listfile=[output list file]");

  if (argc <= 1) {
    FL_LOG(fl::INFO) << gflags::ProgramUsage();
    return -1;
  }

  gflags::ParseCommandLineFlags(&argc, &argv, false);

  SoundLoaderRandomWithoutReplacement soundLoader(
      ListFileReader(FLAGS_input_listfiles, FLAGS_input_rootdir));
  ListFileWriter listFileWriter(FLAGS_output_listfile, FLAGS_output_rootdir);
  std::shared_ptr<SoundEffect> soundEffect =
      createSoundEffect(readSoundEffectConfigFile(FLAGS_sfx_config_filename));

  soundEffect->setEnableCountdown(FLAGS_sfx_start_update * FLAGS_batchsize);

  for (int i = 0; i < soundLoader.size(); ++i) {
    Sound sound = soundLoader.loadIndex(i);
    Sound augmented = soundEffect->apply(sound);

    // if sound.listFileEntry_.audioFilePath_ is an absolute path then make it
    // into relative path
    if (augmented.listFileEntry_.audioFilePath_[0] == pathSeperator()[0]) {
      augmented.listFileEntry_.audioFilePath_.erase(0, 1);
    }
    augmented.listFileEntry_.audioFilePath_ = pathsConcat(
        FLAGS_output_rootdir, augmented.listFileEntry_.audioFilePath_);

    try {
      augmented.writeToFile();
    } catch (std::exception& ex1) {
      FL_LOG(fl::ERROR) << "Fails to write file for sound={"
                        << sound.prettyString() << "} with error={"
                        << ex1.what() << '}';
      continue;
    }

    try {
      listFileWriter.write(augmented.listFileEntry_);
    } catch (std::exception& ex2) {
      FL_LOG(fl::ERROR) << "listFileWriter fails add file="
                        << sound.listFileEntry_.audioFilePath_
                        << " with error=" << ex2.what();
    }
  }
}
