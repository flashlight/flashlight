/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/pkg/text/data/TextDataset.h"
#include "flashlight/fl/common/Init.h"
#include "flashlight/lib/text/dictionary/Defines.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/tokenizer/PartialFileReader.h"
#include "flashlight/lib/text/tokenizer/Tokenizer.h"

using fl::lib::pathsConcat;
using namespace fl::lib;
using namespace fl::lib::text;
using namespace fl::app::lm;

std::string dataDir = "";

Dictionary createDictionary(const std::string& path) {
  Dictionary dictionary;
  auto stream = createInputStream(path);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty()) {
      continue;
    }
    auto tkns = splitOnWhitespace(line, true);
    dictionary.addEntry(tkns.front());
  }
  if (!dictionary.isContiguous()) {
    throw std::runtime_error("Invalid dictionary_ format - not contiguous");
  }
  dictionary.setDefaultIndex(dictionary.getIndex(fl::lib::text::kUnkToken));
  return dictionary;
}

TEST(TextDatasetTest, NoneMode) {
  fl::lib::text::Tokenizer tokenizer;
  fl::lib::text::PartialFileReader partialFileReader(0, 1);
  Dictionary dictionary =
      createDictionary(pathsConcat(dataDir, "dictionary.txt"));

  int tokensPerSample = 5;
  int batchSize = 2;

  TextDataset dataset(
      dataDir,
      "train.txt",
      partialFileReader,
      tokenizer,
      dictionary,
      tokensPerSample,
      batchSize,
      "none");

  ASSERT_EQ(dataset.size(), 4);
  for (int i = 0; i < dataset.size(); i++) {
    auto sample = dataset.get(i);
    ASSERT_EQ(sample.size(), 1);
    ASSERT_EQ(sample[0].dims(0), tokensPerSample);
    ASSERT_EQ(sample[0].dims(1), batchSize);
  }
}

TEST(TextDatasetTest, EosMode) {
  fl::lib::text::Tokenizer tokenizer;
  fl::lib::text::PartialFileReader partialFileReader(0, 1);
  Dictionary dictionary =
      createDictionary(pathsConcat(dataDir, "dictionary.txt"));

  int tokensPerSample = 5;
  int batchSize = 2;

  TextDataset dataset(
      dataDir,
      "train.txt",
      partialFileReader,
      tokenizer,
      dictionary,
      tokensPerSample,
      batchSize,
      "eos");

  ASSERT_EQ(dataset.size(), 4);

  std::vector<int> targetLen = {7, 5, 5, 7};
  for (int i = 0; i < dataset.size(); i++) {
    auto sample = dataset.get(i);
    ASSERT_EQ(sample.size(), 1);
    ASSERT_EQ(sample[0].dims(0), targetLen[i]);
    ASSERT_EQ(sample[0].dims(1), batchSize);
  }
}

TEST(TextDatasetTest, EosModeWithDynamicBatching) {
  fl::lib::text::Tokenizer tokenizer;
  fl::lib::text::PartialFileReader partialFileReader(
      fl::getWorldRank(), fl::getWorldSize());
  Dictionary dictionary =
      createDictionary(pathsConcat(dataDir, "dictionary.txt"));

  int tokensPerSample = 15;

  TextDataset dataset(
      dataDir,
      "train.txt",
      partialFileReader,
      tokenizer,
      dictionary,
      tokensPerSample,
      1,
      "eos",
      true);

  ASSERT_EQ(dataset.size(), 4);

  std::vector<int> targetLen = {5, 6, 7, 7};
  std::vector<int> targetBsz = {3, 2, 2, 1};
  for (int i = 0; i < dataset.size(); i++) {
    auto sample = dataset.get(i);
    ASSERT_EQ(sample.size(), 1);
    ASSERT_EQ(sample[0].dims(0), targetLen[i]);
    ASSERT_EQ(sample[0].dims(1), targetBsz[i]);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();

#ifdef TEXTDATASET_TEST_DATADIR
  dataDir = TEXTDATASET_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
