/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/experimental/soundeffect/ListFile.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "gmock/gmock.h"

#include "flashlight/app/asr/experimental/soundeffect/SoundEffect.h"
#include "flashlight/app/asr/experimental/soundeffect/test/SoundTestUtil.h"

using ::fl::app::asr::sfx::ListFileEntry;
using ::fl::app::asr::sfx::ListFileReader;
using ::fl::app::asr::sfx::ListFileWriter;
using ::fl::app::asr::sfx::testFilename;
using ::testing::Le;

namespace {
std::string loadPath = "/tmp";
}

TEST(ListFileReaderAndWriter, WriteFileAndReadBack) {
  const int nFiles = 100;
  const std::string listFileName = testFilename("sinwave.lst");
  std::vector<std::string> filenames(nFiles, "/dataset/file-number-");
  {
    ListFileWriter writer(listFileName);
    std::cout << "writer=" << writer.prettyString() << std::endl;
    for (int i = 0; i < filenames.size(); ++i) {
      filenames[i] += std::to_string(i);

      ListFileEntry entry;
      entry.sampleId_ = std::to_string(i);
      entry.audioFilePath_ = filenames[i];
      entry.audioSize_ = i;
      entry.transcript_ = {"the good the bad and the ugly"};

      writer.write(entry);
    }
  }

  ListFileReader reader(listFileName);
  std::cout << "reader=" << reader.prettyString() << std::endl;

  for (int i = 0; i < filenames.size(); ++i) {
    const ListFileEntry& entry = reader.read(i);
    std::cout << "entry={" << entry.prettyString() << "}" << std::endl;
    EXPECT_EQ(std::to_string(i), entry.sampleId_);
    EXPECT_EQ(filenames[i], entry.audioFilePath_);
    EXPECT_EQ(i, entry.audioSize_);
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
