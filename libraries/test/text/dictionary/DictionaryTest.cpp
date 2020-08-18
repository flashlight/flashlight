/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/libraries/common/System.h"
#include "flashlight/libraries/text/dictionary/Utils.h"

using fl::lib::pathsConcat;
using namespace fl::lib::text;

std::string loadPath = "";

TEST(DictionaryTest, TestBasic) {
  Dictionary dict;
  dict.addEntry("1", 1);
  dict.addEntry("2", 2);
  dict.addEntry("3", 3);
  dict.addEntry("4", 3);

  ASSERT_EQ(dict.getEntry(1), "1");
  ASSERT_EQ(dict.getEntry(3), "3");

  ASSERT_EQ(dict.getIndex("2"), 2);
  ASSERT_EQ(dict.getIndex("4"), 3);

  ASSERT_EQ(dict.entrySize(), 4);
  ASSERT_EQ(dict.indexSize(), 3);

  dict.addEntry("5");
  ASSERT_EQ(dict.getIndex("5"), 4);
  ASSERT_EQ(dict.entrySize(), 5);

  dict.addEntry("6");
  ASSERT_EQ(dict.getIndex("6"), 5);
  ASSERT_EQ(dict.indexSize(), 5);
}

TEST(DictionaryTest, FromFile) {
  ASSERT_THROW(Dictionary("not_a_real_file"), std::runtime_error);

  Dictionary dict(pathsConcat(loadPath, "test.dict"));
  ASSERT_EQ(dict.entrySize(), 10);
  ASSERT_EQ(dict.indexSize(), 7);
  ASSERT_TRUE(dict.contains("a"));
  ASSERT_FALSE(dict.contains("q"));
  ASSERT_EQ(dict.getEntry(1), "b");
  ASSERT_EQ(dict.getIndex("e"), 4);
}

TEST(DictionaryTest, Dictionary) {
  Dictionary dict;
  dict.addEntry("1", 1);
  dict.addEntry("2", 2);
  dict.addEntry("3", 3);
  dict.addEntry("4", 3);

  ASSERT_EQ(dict.getEntry(1), "1");
  ASSERT_EQ(dict.getEntry(3), "3");

  ASSERT_EQ(dict.getIndex("2"), 2);
  ASSERT_EQ(dict.getIndex("4"), 3);

  ASSERT_EQ(dict.entrySize(), 4);
  ASSERT_EQ(dict.indexSize(), 3);

  dict.addEntry("5");
  ASSERT_EQ(dict.getIndex("5"), 4);
  ASSERT_EQ(dict.entrySize(), 5);

  dict.addEntry("6");
  ASSERT_EQ(dict.getIndex("6"), 5);
  ASSERT_EQ(dict.indexSize(), 5);
}

TEST(DictionaryTest, PackReplabels) {
  Dictionary dict;
  dict.addEntry("1", 1);
  dict.addEntry("2", 2);
  dict.addEntry("3", 3);

  std::vector<int> labels = {5, 6, 6, 6, 10, 8, 8, 10, 10, 10, 10, 10};
  std::vector<std::vector<int>> packedCheck(4);
  packedCheck[0] = {5, 6, 6, 6, 10, 8, 8, 10, 10, 10, 10, 10};
  packedCheck[1] = {5, 6, 1, 6, 10, 8, 1, 10, 1, 10, 1, 10};
  packedCheck[2] = {5, 6, 2, 10, 8, 1, 10, 2, 10, 1};
  packedCheck[3] = {5, 6, 2, 10, 8, 1, 10, 3, 10};

  for (int i = 0; i <= 3; ++i) {
    auto packed = packReplabels(labels, dict, i);
    ASSERT_EQ(packed, packedCheck[i]);
    auto unpacked = unpackReplabels(packed, dict, i);
    ASSERT_EQ(unpacked, labels);
  }
}

TEST(DictionaryTest, UnpackReplabels) {
  Dictionary dict;
  dict.addEntry("1", 1);
  dict.addEntry("2", 2);
  dict.addEntry("3", 3);
  std::vector<int> labels = {6, 3, 7, 2, 8, 0, 1};

  auto unpacked1 = unpackReplabels(labels, dict, 1);
  ASSERT_THAT(unpacked1, ::testing::ElementsAre(6, 3, 7, 2, 8, 0, 0));

  auto unpacked2 = unpackReplabels(labels, dict, 2);
  ASSERT_THAT(unpacked2, ::testing::ElementsAre(6, 3, 7, 7, 7, 8, 0, 0));

  auto unpacked3 = unpackReplabels(labels, dict, 3);
  ASSERT_THAT(unpacked3, ::testing::ElementsAre(6, 6, 6, 6, 7, 7, 7, 8, 0, 0));
}

TEST(DictionaryTest, UnpackReplabelsIgnoresInvalid) {
  Dictionary dict;
  dict.addEntry("1", 1);
  dict.addEntry("2", 2);

  // The initial replabel "1", with no prior token to repeat, is ignored.
  std::vector<int> labels1 = {1, 5, 1, 6};
  auto unpacked1 = unpackReplabels(labels1, dict, 2);
  ASSERT_THAT(unpacked1, ::testing::ElementsAre(5, 5, 6));

  // The final replabel "2", whose prior token is a replabel, is ignored.
  std::vector<int> labels2 = {1, 5, 1, 2, 6};
  auto unpacked2 = unpackReplabels(labels2, dict, 2);
  ASSERT_THAT(unpacked2, ::testing::ElementsAre(5, 5, 6));
  // With maxReps=1, "2" is not considered a replabel, altering the result.
  auto unpacked2_1 = unpackReplabels(labels2, dict, 1);
  ASSERT_THAT(unpacked2_1, ::testing::ElementsAre(5, 5, 2, 6));

  // All replabels past the first "1" are ignored here.
  std::vector<int> labels3 = {5, 1, 2, 1, 2, 6};
  auto unpacked3 = unpackReplabels(labels3, dict, 2);
  ASSERT_THAT(unpacked3, ::testing::ElementsAre(5, 5, 6));
}

TEST(DictionaryTest, UT8Split) {
  // ASCII
  std::string in1 = "Vendetta";
  auto in1Tkns = splitWrd(in1);
  for (int i = 0; i < in1.size(); ++i) {
    ASSERT_EQ(std::string(1, in1[i]), in1Tkns[i]);
  }

  // NFKC encoding
  // @lint-ignore TXT5 Source code should only include printable US-ASCII bytes.
  std::string in2 = "Beyoncé";
  auto in2Tkns = splitWrd(in2);

  // @lint-ignore TXT5 Source code should only include printable US-ASCII bytes.
  std::vector<std::string> in2TknsExp = {"B", "e", "y", "o", "n", "c", "é"};
  ASSERT_EQ(in2Tkns.size(), 7);
  for (int i = 0; i < in2Tkns.size(); ++i) {
    ASSERT_EQ(in2TknsExp[i], in2Tkns[i]);
  }

  // NFKD encoding
  // @lint-ignore TXT5 Source code should only include printable US-ASCII bytes.
  std::string in3 = "Beyoncé";
  auto in3Tkns = splitWrd(in3);
  std::vector<std::string> in3TknsExp = {
      "B", "e", "y", "o", "n", "c", "e", u8"\u0301"};
  ASSERT_EQ(in3Tkns.size(), 8);
  for (int i = 0; i < in3Tkns.size(); ++i) {
    ASSERT_EQ(in3TknsExp[i], in3Tkns[i]);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

// Resolve directory for sample dictionary
#ifdef DICTIONARY_TEST_DATADIR
  loadPath = DICTIONARY_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
