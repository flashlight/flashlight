/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/fl/flashlight.h"

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/pkg/speech/audio/feature/SpeechUtils.h"
#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/data/FeatureTransforms.h"
#include "flashlight/pkg/speech/data/ListFileDataset.h"
#include "flashlight/pkg/speech/data/Utils.h"
#include "flashlight/pkg/speech/decoder/TranscriptionUtils.h"

using namespace fl;
using namespace fl::lib::audio;
using namespace fl::lib::text;
using namespace fl::pkg::speech;

namespace {
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

} // namespace
TEST(FeaturizationTest, AfMatmulCompare) {
  int numTests = 1000;
  while (numTests--) {
    int m = (rand() % 64) + 1;
    int n = (rand() % 128) + 1;
    int k = (rand() % 256) + 1;
    // Note: Arrayfire is column major
    Tensor a = fl::rand({k, m});
    Tensor b = fl::rand({n, k});
    Tensor c = fl::transpose(
        fl::matmul(a, b, MatrixProperty::Transpose, MatrixProperty::Transpose));
    auto aVec = a.toHostVector<float>();
    auto bVec = b.toHostVector<float>();
    auto cVec = cblasGemm(aVec, bVec, n, k);
    ASSERT_TRUE(compareVec(cVec, c.toHostVector<float>(), 1E-4));
  }
}

TEST(FeaturizationTest, Normalize) {
  double threshold = 0.01;
  auto afNormalize = [threshold](const Tensor& in, int batchdim) {
    int64_t elementsPerBatch = in.elements() / in.dim(batchdim);
    auto in2d = fl::reshape(in, {elementsPerBatch, in.dim(batchdim)});

    Tensor meandiff =
        (in2d -
         fl::tile(
             fl::mean(in2d, {0}, /* keepDims = */ true), {elementsPerBatch}));

    Tensor stddev = fl::std(in2d, {0}, /* keepDims = */ true);
    stddev = fl::where(stddev > threshold, stddev, 1.0);

    return fl::reshape(
        meandiff / fl::tile(stddev, {elementsPerBatch}), in.shape());
  };
  auto arr = fl::rand({13, 17, 19});
  auto arrVec = arr.toHostVector<float>();

  auto arrVecNrm = normalize(arrVec, 19, threshold);
  auto arrNrm =
      Tensor::fromBuffer(arr.shape(), arrVecNrm.data(), MemoryLocation::Host);
  ASSERT_TRUE(
      fl::all(fl::abs(arrNrm - afNormalize(arr, 2)) < 1E-5).asScalar<bool>());
}

TEST(FeaturizationTest, Transpose) {
  auto arr = fl::rand({13, 17, 19, 23});
  auto arrVec = arr.toHostVector<float>();
  auto arrVecT = transpose2d<float>(arrVec, 17, 13, 19 * 23);
  auto arrT = Tensor::fromVector({17, 13, 19, 23}, arrVecT);
  ASSERT_TRUE(
      fl::all(arrT - fl::transpose(arr, {1, 0, 2, 3}) == 0.0).asScalar<bool>());
}

TEST(FeaturizationTest, localNormalize) {
  auto afNormalize = [](const Tensor& in, int64_t lw, int64_t rw) {
    auto out = in;
    for (int64_t b = 0; b < in.dim(3); ++b) {
      for (int64_t i = 0; i < in.dim(0); ++i) {
        int64_t b_idx = (i - lw > 0) ? (i - lw) : 0;
        int64_t e_idx = (in.dim(0) - 1 > i + rw) ? (i + rw) : (in.dim(0) - 1);

        Tensor slice = in(fl::range(b_idx, e_idx + 1), fl::span, fl::span, b);
        auto mean = fl::mean(slice).scalar<float>();
        auto stddev = fl::std(slice).scalar<float>();

        out(i, fl::span, fl::span, b) -= mean;
        if (stddev > 0.0) {
          out(i, fl::span, fl::span, b) /= stddev;
        }
      }
    }
    return out;
  };
  auto arr = fl::rand({47, 67, 2, 10}); // FRAMES X FEAT X CHANNELS X BATCHSIZE
  auto arrVec = arr.toHostVector<float>();

  std::vector<std::pair<int, int>> ctx = {
      {0, 0}, {1, 1}, {2, 2}, {4, 4}, {1024, 1024}, {10, 0}, {2, 12}};

  for (auto c : ctx) {
    auto arrVecNrm = localNormalize(
        arrVec,
        c.first /* context */,
        c.second,
        arr.dim(0) /* frames */,
        arr.dim(3) /*batches */);
    auto arrNrm =
        Tensor::fromBuffer(arr.shape(), arrVecNrm.data(), MemoryLocation::Host);
    ASSERT_TRUE(
        fl::all(fl::abs(arrNrm - afNormalize(arr, c.first, c.second)) < 1E-4)
            .asScalar<bool>());
  }
}

TEST(FeaturizationTest, TargetTknTestStandaloneSep) {
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
      false);

  std::vector<std::string> resT = {
      "ab", "cd", "ef", "||", "ab", "cd", "||", "t", "r", "||"};
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
      false);

  std::vector<std::string> resT2 = {
      "ab", "cd", "ef", "||", "ab", "cd", "||", "||", "t", "r"};
  ASSERT_EQ(res2.size(), resT2.size());
  for (int index = 0; index < res2.size(); index++) {
    ASSERT_EQ(res2[index], resT2[index]);
  }
}

TEST(FeaturizationTest, TargetTknTestInsideSep) {
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
      false);

  std::vector<std::string> resT = {
      "_", "a", "f", "f", "_hel", "lo", "_ma", "ma", "_", "a", "f"};
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
      false);

  std::vector<std::string> resT2 = {
      "a", "f", "f", "_", "_hel", "lo", "_ma", "ma", "_", "a", "f"};
  ASSERT_EQ(res.size(), resT2.size());
  for (int index = 0; index < res2.size(); index++) {
    ASSERT_EQ(res2[index], resT2[index]);
  }
}

TEST(FeaturizationTest, WrdToTarget) {
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

  // NOTE: word separator has no effect when fallback2Ltr is false
  std::vector<std::string> words = {"123", "456"};
  auto target = wrd2Target(words, lexicon, dict, "", 0, false, false, false);
  ASSERT_THAT(target, ::testing::ElementsAreArray({"1", "23_", "456_"}));

  std::vector<std::string> words1 = {"789", "010"};
  auto target1 = wrd2Target(words1, lexicon, dict, "_", 0, false, false, false);
  ASSERT_THAT(target1, ::testing::ElementsAreArray({"_7", "89", "_0", "10"}));

  std::vector<std::string> words2 = {"105", "2100"};
  auto target2 = wrd2Target(words2, lexicon, dict, "", 0, false, false, false);
  ASSERT_THAT(
      target2, ::testing::ElementsAreArray({"10", "5", "2", "1", "00"}));

  std::vector<std::string> words3 = {"12", "888", "12"};
  auto target3 = wrd2Target(words3, lexicon, dict, "_", 0, false, false, false);
  ASSERT_THAT(
      target3,
      ::testing::ElementsAreArray(
          {"1", "2", "_", "8", "8", "8", "_", "1", "2", "_"}));

  // unknown words "111", "199"
  std::vector<std::string> words4 = {"111", "789", "199"};
  // fall back to letters, wordsep to left and skip unknown
  auto target4 = wrd2Target(words4, lexicon, dict, "_", 0, true, false, true);
  ASSERT_THAT(
      target4,
      ::testing::ElementsAreArray({"_", "1", "1", "1", "_7", "89", "_", "1"}));
  // fall back to letters, wordsep to right and skip unknown
  target4 = wrd2Target(words4, lexicon, dict, "_", 0, false, true, true);
  ASSERT_THAT(
      target4,
      ::testing::ElementsAreArray({"1", "1", "1", "_", "_7", "89", "1", "_"}));

  // skip unknown
  target4 = wrd2Target(words4, lexicon, dict, "", 0, false, false, true);
  ASSERT_THAT(target4, ::testing::ElementsAreArray({"_7", "89"}));
}

TEST(FeaturizationTest, TargetToSingleLtr) {
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

TEST(FeaturizationTest, inputFeaturizer) {
  auto channels = 2;
  auto samplerate = 16000;
  FeatureParams featParams(
      samplerate,
      25, // framesize
      10, // framestride
      40, // filterbanks
      0, // lowfreqfilterbank,
      samplerate / 2, // highfreqfilterbank
      -1, // mfcccoeffs
      kLifterParam, // lifterparam
      0, // delta window
      0 // delta-delta window
  );
  featParams.useEnergy = false;
  featParams.usePower = false;
  featParams.zeroMeanFrame = false;
  auto inputFeaturizerRaw =
      inputFeatures(featParams, FeatureType::NONE, {-1, -1}, {});
  auto inputFeaturizerMfsc =
      inputFeatures(featParams, FeatureType::MFSC, {-1, -1}, {});
  for (int size = 1; size < 10; ++size) {
    std::vector<float> input(size * samplerate * channels);
    for (int j = 0; j < input.size(); ++j) {
      // channel 1 is same as channel 2
      input[j] = std::sin(2 * M_PI * (j / 2) / samplerate);
    }

    int insize = size * samplerate;
    auto inArray =
        inputFeaturizerRaw(input.data(), {channels, insize}, fl::dtype::f32);
    ASSERT_EQ(inArray.shape(), Shape({insize, 1, channels}));
    Tensor ch1 = inArray(fl::span, fl::span, 0);
    Tensor ch2 = inArray(fl::span, fl::span, 1);
    ASSERT_TRUE(fl::amax(fl::abs(ch1 - ch2)).scalar<float>() < 1E-5);

    inArray =
        inputFeaturizerMfsc(input.data(), {channels, insize}, fl::dtype::f32);
    auto nFrames = 1 + (insize - 25 * 16) / (10 * 16);
    ASSERT_EQ(inArray.shape(), Shape({nFrames, 40, channels}));
    ch1 = inArray(fl::span, fl::span, 0, fl::span);
    ch2 = inArray(fl::span, fl::span, 1, fl::span);
    ASSERT_TRUE(fl::amax(fl::abs(ch1 - ch2)).scalar<float>() < 1E-5);
  }
}

TEST(FeaturizationTest, targetFeaturizer) {
  using fl::pkg::speech::kEosToken;

  auto tokenDict = getDict();
  tokenDict.addEntry(kEosToken);
  auto lexicon = getLexicon();
  std::vector<std::vector<char>> targets = {
      {'a', 'b', 'c', 'c', 'c'}, {'b', 'c', 'd', 'd'}};

  TargetGenerationConfig targetGenConfig(
      "",
      0,
      kCtcCriterion,
      "",
      false,
      0,
      true /* skip unk */,
      false /* fallback2LetterWordSepLeft */,
      true /* fallback2LetterWordSepLeft */);

  auto targetFeaturizer = targetFeatures(tokenDict, lexicon, targetGenConfig);

  auto tgtArray = targetFeaturizer(
      targets[0].data(),
      {static_cast<long long>(targets[0].size())},
      fl::dtype::b8);
  int tgtLen = 5;
  ASSERT_EQ(tgtArray.shape(), Shape({tgtLen}));
  ASSERT_EQ(tgtArray.type(), fl::dtype::s32);
  std::vector<int> tgtArrayVec(tgtLen);
  tgtArray.host(tgtArrayVec.data());

  ASSERT_EQ(tgtArrayVec[0], 0);
  ASSERT_EQ(tgtArrayVec[1], 1);
  ASSERT_EQ(tgtArrayVec[2], 2);
  ASSERT_EQ(tgtArrayVec[3], 2);
  ASSERT_EQ(tgtArrayVec[4], 2);

  auto targetGenConfigEos = TargetGenerationConfig(
      "",
      0,
      kCtcCriterion,
      "",
      true, // changed from above
      0,
      true /* skip unk */,
      false /* fallback2LetterWordSepLeft */,
      true /* fallback2LetterWordSepLeft */);
  targetFeaturizer = targetFeatures(tokenDict, lexicon, targetGenConfigEos);
  tgtArray = targetFeaturizer(
      targets[1].data(),
      {static_cast<long long>(targets[1].size())},
      fl::dtype::b8);
  tgtLen = 5;
  int eosIdx = tokenDict.getIndex(kEosToken);
  ASSERT_EQ(tgtArray.shape(), Shape({tgtLen}));
  ASSERT_EQ(tgtArray.type(), fl::dtype::s32);
  tgtArray.host(tgtArrayVec.data());
  ASSERT_EQ(tgtArrayVec[0], 1);
  ASSERT_EQ(tgtArrayVec[1], 2);
  ASSERT_EQ(tgtArrayVec[2], 3);
  ASSERT_EQ(tgtArrayVec[3], 3);
  ASSERT_EQ(tgtArrayVec[4], eosIdx);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
