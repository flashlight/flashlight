/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arrayfire.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flashlight/fl/flashlight.h"

#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/data/Featurize.h"
#include "flashlight/app/asr/data/Utils.h"
#include "flashlight/app/asr/data/ListFilesDataset.h"

#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

using namespace fl::lib;
using namespace fl::app::asr;
using fl::lib::text::Dictionary;
using fl::lib::text::DictionaryMap;
using fl::lib::text::kUnkToken;
using fl::lib::text::LexiconMap;

namespace {

std::string loadPath = "";

Dictionary getDict() {
  Dictionary dict;
  std::string ltr = "a";
  int alphabet_sz = 26;
  while (alphabet_sz--) {
    dict.addEntry(ltr);
    ltr[0] += 1;
  }
  dict.addEntry("|");
  dict.addEntry("'");
  dict.addEntry("L", dict.getIndex("|"));
  dict.addEntry("N", dict.getIndex("|"));
  return dict;
}

LexiconMap getLexicon() {
  LexiconMap lexicon;
  lexicon["uh"].push_back({"u", "h"});
  lexicon["oh"].push_back({"o", "h"});
  lexicon[kUnkToken] = {};
  return lexicon;
}

std::vector<std::string> loadTarget(const std::string& filepath) {
  std::vector<std::string> tokens;
  std::ifstream infile(filepath);
  if (!infile) {
    throw std::runtime_error(
        std::string() + "Could not read file '" + filepath + "'");
  }
  std::string line;
  while (std::getline(infile, line)) {
    auto tknsStr = splitOnWhitespace(line, true);
    for (const auto& tkn : tknsStr) {
      tokens.emplace_back(tkn);
    }
  }
  return tokens;
}

} // namespace

TEST(DataTest, inputFeaturizer) {
  auto dict = getDict();
  auto inputFeaturizer = [](std::vector<std::vector<float>> in,
                            const Dictionary& d) {
    std::vector<LoaderData> data;
    for (const auto& i : in) {
      data.emplace_back();
      data.back().input = i;
    }

    DictionaryMap dicts;
    dicts.insert({kTargetIdx, d});

    auto feat = featurize(data, dicts);
    return af::array(feat.inputDims, feat.input.data());
  };

  std::vector<std::vector<float>> inputs;
  gflags::FlagSaver flagsaver;
  FLAGS_channels = 2;
  FLAGS_samplerate = 16000;
  for (int i = 0; i < 10; ++i) {
    inputs.emplace_back(i * FLAGS_samplerate * FLAGS_channels);
    for (int j = 0; j < inputs.back().size(); ++j) {
      inputs.back()[j] = std::sin(2 * M_PI * (j / 2) / FLAGS_samplerate);
    }
  }
  auto inArray = inputFeaturizer(inputs, dict);
  ASSERT_EQ(
      inArray.dims(), af::dim4(9 * FLAGS_samplerate, FLAGS_channels, 1, 10));
  af::array ch1 = inArray(af::span, 0, af::span);
  af::array ch2 = inArray(af::span, 1, af::span);
  ASSERT_TRUE(af::max<double>(af::abs(ch1 - ch2)) < 1E-5);

  FLAGS_mfsc = true;
  inArray = inputFeaturizer(inputs, dict);
  auto nFrames = 1 + (9 * FLAGS_samplerate - 25 * 16) / (10 * 16);
  ASSERT_EQ(inArray.dims(), af::dim4(nFrames, 40, FLAGS_channels, 10));
  ch1 = inArray(af::span, af::span, 0, af::span);
  ch2 = inArray(af::span, af::span, 1, af::span);
  ASSERT_TRUE(af::max<double>(af::abs(ch1 - ch2)) < 1E-5);
}

TEST(DataTest, targetFeaturizer) {
  auto dict = getDict();
  dict.addEntry(kEosToken);
  std::vector<std::vector<std::string>> targets = {{"a", "b", "c", "c", "c"},
                                                   {"b", "c", "d", "d"}};

  gflags::FlagSaver flagsaver;
  FLAGS_replabel = 0;
  FLAGS_criterion = kCtcCriterion;

  auto targetFeaturizer = [](std::vector<std::vector<std::string>> tgt,
                             const Dictionary& d) {
    std::vector<LoaderData> data;
    for (const auto& t : tgt) {
      data.emplace_back();
      data.back().targets[kTargetIdx] = t;
    }

    DictionaryMap dicts;
    dicts.insert({kTargetIdx, d});

    auto feat = featurize(data, dicts);
    return af::array(
        feat.targetDims[kTargetIdx], feat.targets[kTargetIdx].data());
  };

  auto tgtArray = targetFeaturizer(targets, dict);
  int tgtLen = 5;
  ASSERT_EQ(tgtArray.dims(0), tgtLen);
  ASSERT_EQ(tgtArray(tgtLen - 1, 0).scalar<int>(), 2);
  ASSERT_EQ(tgtArray(tgtLen - 1, 1).scalar<int>(), kTargetPadValue);
  ASSERT_EQ(tgtArray(tgtLen - 2, 1).scalar<int>(), 3);

  FLAGS_eostoken = true;
  tgtArray = targetFeaturizer(targets, dict);
  tgtLen = 6;
  int eosIdx = dict.getIndex(kEosToken);
  ASSERT_EQ(tgtArray.dims(0), tgtLen);
  ASSERT_EQ(tgtArray(tgtLen - 1, 0).scalar<int>(), eosIdx);
  ASSERT_EQ(tgtArray(tgtLen - 1, 1).scalar<int>(), eosIdx);
  ASSERT_EQ(tgtArray(tgtLen - 2, 1).scalar<int>(), eosIdx);
}

TEST(RoundRobinBatchShufflerTest, params) {
  auto packer = RoundRobinBatchPacker(2, 2, 0);
  auto batches = packer.getBatches(11, 0);
  EXPECT_EQ(batches.size(), 3);
  ASSERT_THAT(batches[0], ::testing::ElementsAre(8, 9));
  ASSERT_THAT(batches[1], ::testing::ElementsAre(0, 1));
  ASSERT_THAT(batches[2], ::testing::ElementsAre(4, 5));

  packer = RoundRobinBatchPacker(2, 2, 1);
  batches = packer.getBatches(11, 0);
  EXPECT_EQ(batches.size(), 3);
  ASSERT_THAT(batches[0], ::testing::ElementsAre(10));
  ASSERT_THAT(batches[1], ::testing::ElementsAre(2, 3));
  ASSERT_THAT(batches[2], ::testing::ElementsAre(6, 7));

  // No shuffling
  packer = RoundRobinBatchPacker(2, 2, 0);
  batches = packer.getBatches(11, -1);
  EXPECT_EQ(batches.size(), 3);
  ASSERT_THAT(batches[0], ::testing::ElementsAre(0, 1));
  ASSERT_THAT(batches[1], ::testing::ElementsAre(4, 5));
  ASSERT_THAT(batches[2], ::testing::ElementsAre(8, 9));

  batches = packer.getBatches(10, -1);
  EXPECT_EQ(batches.size(), 3);
  ASSERT_THAT(batches[0], ::testing::ElementsAre(0, 1));
  ASSERT_THAT(batches[1], ::testing::ElementsAre(4, 5));
  ASSERT_THAT(batches[2], ::testing::ElementsAre(8));

  batches = packer.getBatches(9, -1);
  EXPECT_EQ(batches.size(), 2);
  ASSERT_THAT(batches[0], ::testing::ElementsAre(0, 1));
  ASSERT_THAT(batches[1], ::testing::ElementsAre(4, 5));

  batches = packer.getBatches(8, -1);
  EXPECT_EQ(batches.size(), 2);
  ASSERT_THAT(batches[0], ::testing::ElementsAre(0, 1));
  ASSERT_THAT(batches[1], ::testing::ElementsAre(4, 5));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

// Resolve directory for data
#ifdef DATA_TEST_DATADIR
  loadPath = DATA_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
