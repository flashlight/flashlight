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
#include "flashlight/app/asr/data/ListFilesDataset.h"
#include "flashlight/app/asr/decoder/TranscriptionUtils.h"

#include "flashlight/lib/audio/feature/SpeechUtils.h"

using namespace fl;
using namespace fl::lib::audio;
using namespace fl::lib::text;
using namespace fl::app::asr;

template <typename T>
bool compareVec(std::vector<T> A, std::vector<T> B, float precision = 1E-5) {
  if (A.size() != B.size()) {
    return false;
  }
  for (std::size_t i = 0; i < A.size(); ++i) {
    if (std::abs(A[i] - B[i]) > precision) {
      return false;
    }
  }
  return true;
}

TEST(FeatureTest, AfMatmulCompare) {
  int numTests = 1000;
  auto afToVec = [](const af::array& arr) {
    std::vector<float> vec(arr.elements());
    arr.host(vec.data());
    return vec;
  };
  while (numTests--) {
    int m = (rand() % 64) + 1;
    int n = (rand() % 128) + 1;
    int k = (rand() % 256) + 1;
    // Note: Arrayfire is column major
    af::array a = af::randu(k, m);
    af::array b = af::randu(n, k);
    af::array c = af::matmul(a, b, AF_MAT_TRANS, AF_MAT_TRANS).T();
    auto aVec = afToVec(a);
    auto bVec = afToVec(b);
    auto cVec = cblasGemm(aVec, bVec, n, k);
    ASSERT_TRUE(compareVec(cVec, afToVec(c), 1E-4));
  }
}

TEST(FeatureTest, Normalize) {
  double threshold = 0.01;
  auto afNormalize = [threshold](const af::array& in, int batchdim) {
    auto elementsPerBatch = in.elements() / in.dims(batchdim);
    auto in2d = af::moddims(in, elementsPerBatch, in.dims(batchdim));

    af::array meandiff = (in2d - af::tile(af::mean(in2d, 0), elementsPerBatch));

    af::array stddev = af::stdev(in2d, 0);
    af::replace(stddev, stddev > threshold, 1.0);

    return af::moddims(
        meandiff / af::tile(stddev, elementsPerBatch), in.dims());
  };
  auto arr = af::randu(13, 17, 19);
  std::vector<float> arrVec(arr.elements());
  arr.host(arrVec.data());

  auto arrVecNrm = normalize(arrVec, 19, threshold);
  auto arrNrm = af::array(arr.dims(), arrVecNrm.data());
  ASSERT_TRUE(af::allTrue<bool>(af::abs(arrNrm - afNormalize(arr, 2)) < 1E-5));
}

TEST(FeatureTest, Transpose) {
  auto arr = af::randu(13, 17, 19, 23);
  std::vector<float> arrVec(arr.elements());
  arr.host(arrVec.data());
  auto arrVecT = transpose2d<float>(arrVec, 17, 13, 19 * 23);
  auto arrT = af::array(17, 13, 19, 23, arrVecT.data());
  ASSERT_TRUE(af::allTrue<bool>(arrT - arr.T() == 0.0));
}

TEST(FeatureTest, localNormalize) {
  auto afNormalize = [](const af::array& in, int64_t lw, int64_t rw) {
    auto out = in;
    for (int64_t b = 0; b < in.dims(3); ++b) {
      for (int64_t i = 0; i < in.dims(0); ++i) {
        int64_t b_idx = (i - lw > 0) ? (i - lw) : 0;
        int64_t e_idx = (in.dims(0) - 1 > i + rw) ? (i + rw) : (in.dims(0) - 1);

        // Need to call af::flat because of some weird bug in Arrayfire
        af::array slice = in(af::seq(b_idx, e_idx), af::span, af::span, b);
        auto mean = af::mean<float>(af::flat(slice));
        auto stddev = af::stdev<float>(af::flat(slice));

        out(i, af::span, af::span, b) -= mean;
        if (stddev > 0.0) {
          out(i, af::span, af::span, b) /= stddev;
        }
      }
    }
    return out;
  };
  auto arr = af::randu(47, 67, 2, 10); // FRAMES X FEAT X CHANNELS X BATCHSIZE
  std::vector<float> arrVec(arr.elements());
  arr.host(arrVec.data());

  std::vector<std::pair<int, int>> ctx = {
      {0, 0}, {1, 1}, {2, 2}, {4, 4}, {1024, 1024}, {10, 0}, {2, 12}};

  for (auto c : ctx) {
    auto arrVecNrm = localNormalize(
        arrVec,
        c.first /* context */,
        c.second,
        arr.dims(0) /* frames */,
        arr.dims(3) /*batches */);
    auto arrNrm = af::array(arr.dims(), arrVecNrm.data());
    ASSERT_TRUE(af::allTrue<bool>(
        af::abs(arrNrm - afNormalize(arr, c.first, c.second)) < 1E-4));
  }
}

TEST(FeatureTest, TargetTknTestStandaloneSep) {
  Dictionary tokens;
  std::string sep = "||";
  tokens.addEntry("ab");
  tokens.addEntry("cd");
  tokens.addEntry("ef");
  tokens.addEntry("t");
  tokens.addEntry("r");
  tokens.addEntry(sep);

  LexiconMap lexicon;
  lexicon["abcd"].push_back({"ab", "cd", "||"});
  lexicon["abcdef"].push_back({"ab", "cd", "ef", "||"});

  std::vector<std::string> words = {"abcdef", "abcd", "tr"};
  auto res = wrd2Target(
    words,
    lexicon,
    tokens,
    sep,
    0,
    false,
    true, // fallback right
    false
  );

  std::vector<std::string> resT = {"ab", "cd", "ef", "||", "ab", "cd", "||", "t", "r", "||"};
  ASSERT_EQ(res.size(), resT.size());
  for (int index = 0; index < res.size(); index++) {
    ASSERT_EQ(res[index], resT[index]);
  }

  auto res2 = wrd2Target(
    words,
    lexicon,
    tokens,
    sep,
    0,
    true, // fallback left
    false,
    false
  );

  std::vector<std::string> resT2 = {"ab", "cd", "ef", "||", "ab", "cd", "||", "||", "t", "r"};
  ASSERT_EQ(res2.size(), resT2.size());
  for (int index = 0; index < res2.size(); index++) {
    ASSERT_EQ(res2[index], resT2[index]);
  }
}

TEST(FeatureTest, TargetTknTestInsideSep) {
  Dictionary tokens;
  std::string sep = "_";
  tokens.addEntry("_hel");
  tokens.addEntry("lo");
  tokens.addEntry("_ma");
  tokens.addEntry("ma");
  tokens.addEntry(sep);
  tokens.addEntry("f");
  tokens.addEntry("a");

  LexiconMap lexicon;
  lexicon["hello"].push_back({"_hel", "lo"});
  lexicon["mama"].push_back({"_ma", "ma"});
  lexicon["af"].push_back({"_", "a", "f"});

  std::vector<std::string> words = {"aff", "hello", "mama", "af"};
  auto res = wrd2Target(
    words,
    lexicon,
    tokens,
    sep,
    0,
    true, // fallback left
    false,
    false
  );

  std::vector<std::string> resT = {"_", "a", "f", "f", "_hel", "lo", "_ma", "ma", "_", "a", "f"};
  ASSERT_EQ(res.size(), resT.size());
  for (int index = 0; index < res.size(); index++) {
    ASSERT_EQ(res[index], resT[index]);
  }

  auto res2 = wrd2Target(
    words,
    lexicon,
    tokens,
    sep,
    0,
    false,
    true, // fallback right
    false
  );

  std::vector<std::string> resT2 = {"a", "f", "f", "_", "_hel", "lo", "_ma", "ma", "_", "a", "f"};
  ASSERT_EQ(res.size(), resT2.size());
  for (int index = 0; index < res2.size(); index++) {
    ASSERT_EQ(res2[index], resT2[index]);
  }
}

TEST(FeatureTest, WrdToTarget) {
  gflags::FlagSaver flagsaver;
  FLAGS_wordseparator = "_";

  LexiconMap lexicon;
  // word pieces with word separator in the end
  lexicon["123"].push_back({"1", "23_"});
  lexicon["456"].push_back({"456_"});
  // word pieces with word separator in the beginning
  lexicon["789"].push_back({"_7", "89"});
  lexicon["010"].push_back({"_0", "10"});
  // word pieces without word separators
  lexicon["105"].push_back({"10", "5"});
  lexicon["2100"].push_back({"2", "1", "00"});
  // letters
  lexicon["888"].push_back({"8", "8", "8", "_"});
  lexicon["12"].push_back({"1", "2", "_"});
  lexicon[kUnkToken] = {};

  Dictionary dict;
  for (auto l : lexicon) {
    for (auto p : l.second) {
      for (auto c : p) {
        if (!dict.contains(c)) {
          dict.addEntry(c);
        }
      }
    }
  }
  if (!dict.contains(FLAGS_wordseparator)) {
    dict.addEntry(FLAGS_wordseparator);
  }

  std::vector<std::string> words = {"123", "456"};
  auto target = wrd2Target(words, lexicon, dict, false, false);
  ASSERT_THAT(target, ::testing::ElementsAreArray({"1", "23_", "456_"}));

  std::vector<std::string> words1 = {"789", "010"};
  auto target1 = wrd2Target(words1, lexicon, dict, false, false);
  ASSERT_THAT(target1, ::testing::ElementsAreArray({"_7", "89", "_0", "10"}));

  std::vector<std::string> words2 = {"105", "2100"};
  auto target2 = wrd2Target(words2, lexicon, dict, false, false);
  ASSERT_THAT(
      target2, ::testing::ElementsAreArray({"10", "5", "2", "1", "00"}));

  std::vector<std::string> words3 = {"12", "888", "12"};
  auto target3 = wrd2Target(words3, lexicon, dict, false, false);
  ASSERT_THAT(
      target3,
      ::testing::ElementsAreArray(
          {"1", "2", "_", "8", "8", "8", "_", "1", "2", "_"}));

  // unknown words "111", "199"
  std::vector<std::string> words4 = {"111", "789", "199"};
  // fall back to letters and skip unknown, word sep is part of word
  FLAGS_usewordpiece = true;
  auto target4 = wrd2Target(words4, lexicon, dict, true, true);
  ASSERT_THAT(
    target4,
    ::testing::ElementsAreArray({"_", "1", "1", "1", "_7", "89", "_", "1"}));
  // fall back to letters and skip unknown, word sep is separate token
  FLAGS_usewordpiece = false;
  target4 = wrd2Target(words4, lexicon, dict, true, true);
  ASSERT_THAT(
    target4,
    ::testing::ElementsAreArray({"1", "1", "1", "_", "_7", "89", "1", "_"}));

  // skip unknown
  target4 = wrd2Target(words4, lexicon, dict, false, true);
  ASSERT_THAT(target4, ::testing::ElementsAreArray({"_7", "89"}));
}

TEST(FeatureTest, TargetToSingleLtr) {
  std::string wordseparator = "_";
  bool usewordpiece = true;

  Dictionary dict;
  for (int i = 0; i < 10; ++i) {
    dict.addEntry(std::to_string(i), i);
  }
  dict.addEntry("_", 10);
  dict.addEntry("23_", 230);
  dict.addEntry("456_", 4560);

  std::vector<int> words = {1, 230, 4560};
  auto target = tknIdx2Ltr(words, dict, usewordpiece, wordseparator);
  ASSERT_THAT(
      target, ::testing::ElementsAreArray({"1", "2", "3", "_", "4", "5", "6"}));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
