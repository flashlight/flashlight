/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "gmock/gmock.h"

#include "flashlight/app/asr/experimental/soundeffect/ListFile.h"
#include "flashlight/app/asr/experimental/soundeffect/Sound.h"
#include "flashlight/app/asr/experimental/soundeffect/SoundEffect.h"
#include "flashlight/app/asr/experimental/soundeffect/SoundLoader.h"
#include "flashlight/app/asr/experimental/soundeffect/test/SoundTestUtil.h"
#include "flashlight/lib/common/System.h"

using namespace ::fl::app::asr::sfx;
using ::fl::lib::pathsConcat;
using ::testing::FloatNear;
using ::testing::Ge;
using ::testing::Le;
using ::testing::Pointwise;

namespace {
std::string loadPath = "/tmp";
}

template <typename SoundLoaderType>
class WriteListFileAndLoadBackSoundTestFixture : public ::testing::Test {
 public:
  void SetUp() override {
    const std::string signalBaseDir = testFilename("sinwave");
    const std::string listFileName = pathsConcat(signalBaseDir, "sinwave.lst");
    const size_t nSignalFiles = 100;
    const size_t signalSoundLen = 40;
    const size_t signalSoundFreqMin = 1e2;
    const size_t signalSoundFreqMax = 1e2;
    const float signalAmplitude = 0.2;
    filenamesAndSounds_ = writeSinWaveSoundFiles(
        signalBaseDir,
        nSignalFiles,
        signalSoundLen,
        signalAmplitude,
        signalSoundFreqMin,
        signalSoundFreqMax);

    // ListFileWriter flushes and closes the file on destructor.
    {
      ListFileWriter writer(listFileName);
      std::cout << "writer=" << writer.prettyString() << std::endl;
      for (int i = 0; i < filenamesAndSounds_.size(); ++i) {
        ListFileEntry entry;
        entry.sampleId_ = std::to_string(i);
        entry.audioFilePath_ = filenamesAndSounds_[i].first;
        entry.audioSize_ = filenamesAndSounds_[i].second.size();
        entry.transcript_ = {"the good the bad and the ugly"};

        writer.write(entry);
        listFileEntries_.push_back(entry);
      }
    }

    ListFileReader listFileReader(listFileName);
    soundLoader_ = std::make_shared<SoundLoaderType>(listFileReader);
  }

  std::shared_ptr<SoundLoaderType> soundLoader_;
  std::vector<ListFileEntry> listFileEntries_;
  std::vector<std::pair<std::string, Sound>> filenamesAndSounds_;
};

using SoundLoaderTypes = ::testing::Types<
    SoundLoaderRandomWithReplacement,
    SoundLoaderRandomWithoutReplacement>;
TYPED_TEST_SUITE(WriteListFileAndLoadBackSoundTestFixture, SoundLoaderTypes);

TYPED_TEST(
    WriteListFileAndLoadBackSoundTestFixture,
    CompareEachSoundFileByIndex) {
  for (int i = 0; i < this->filenamesAndSounds_.size(); ++i) {
    Sound sound = this->soundLoader_->loadIndex(i);
    EXPECT_EQ(
        sound.listFileEntry_.prettyString(),
        this->listFileEntries_[i].prettyString());
    EXPECT_THAT(
        *sound.getCpuData(),
        Pointwise(
            FloatNear(0.01),
            *this->filenamesAndSounds_[i].second.getCpuData()));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  FL_LOG(fl::INFO) << "logging works!";

// Resolve directory for data
#ifdef DATA_TEST_DATADIR
  loadPath = DATA_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
