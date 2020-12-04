/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/lib/text/tokenizer/PartialFileReader.h"

std::string loadPath = "";

TEST(PartialFileReaderTest, Reading) {
  fl::lib::text::PartialFileReader subFile1(0, 2);
  fl::lib::text::PartialFileReader subFile2(1, 2);

  subFile1.loadFile(fl::lib::pathsConcat(loadPath, "test.txt"));
  subFile2.loadFile(fl::lib::pathsConcat(loadPath, "test.txt"));

  std::vector<std::string> target = {"this",
                                     "is",
                                     "a",
                                     "test",
                                     "just",
                                     "a",
                                     "test",
                                     "for",
                                     "our",
                                     "perfect",
                                     "tokenizer",
                                     "and",
                                     "splitter"};

  std::vector<std::string> loadedWords;
  while (subFile1.hasNextLine()) {
    auto line = subFile1.getLine();
    auto tokens = fl::lib::splitOnWhitespace(line, true);
    loadedWords.insert(loadedWords.end(), tokens.begin(), tokens.end());
  }
  while (subFile2.hasNextLine()) {
    auto line = subFile2.getLine();
    auto tokens = fl::lib::splitOnWhitespace(line);
    loadedWords.insert(loadedWords.end(), tokens.begin(), tokens.end());
  }

  ASSERT_EQ(target.size(), loadedWords.size());
  for (int i = 0; i < target.size(); i++) {
    ASSERT_EQ(target[i], loadedWords[i]);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

#ifdef TOKENIZER_TEST_DATADIR
  loadPath = TOKENIZER_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
