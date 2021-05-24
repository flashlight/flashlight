/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Utils-inl.h"
#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/lib/text/decoder/Decoder.h"
#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/lm/LM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"

namespace fl {
namespace app {
namespace asr {

struct DecodeMasterLexiconFreeOptions {
  int beamSize;
  int beamSizeToken;
  double beamThreshold;
  double lmWeight;
  double silScore;
  bool logAdd;
  std::string silToken;
  std::string blankToken;
};

struct DecodeMasterLexiconOptions {
  int beamSize;
  int beamSizeToken;
  double beamThreshold;
  double lmWeight;
  double silScore;
  double wordScore;
  double unkScore;
  bool logAdd;
  std::string silToken;
  std::string blankToken;
  std::string unkToken;
  fl::lib::text::SmearingMode smearMode;
};

struct DecodeMasterTrainOptions {
  int repLabel;
  bool wordSepIsPartOfToken;
  std::string surround;
  std::string wordSep;
  int32_t targetPadIdx;
};

class DecodeMaster {
 public:
  explicit DecodeMaster(
      const std::shared_ptr<fl::Module> net,
      const std::shared_ptr<fl::lib::text::LM> lm,
      const bool isTokenLM,
      const bool usePlugin,
      const fl::lib::text::Dictionary& tokenDict,
      const fl::lib::text::Dictionary& wordDict,
      const DecodeMasterTrainOptions& trainOpt);

  // compute emissions
  virtual std::shared_ptr<fl::Dataset> forward(
      const std::shared_ptr<fl::Dataset>& ds);

  // decode emissions with an existing decoder
  std::shared_ptr<fl::Dataset> decode(
      const std::shared_ptr<fl::Dataset>& eds,
      fl::lib::text::Decoder& decoder);

  // returns token edit distance and word edit distance stats
  std::pair<std::vector<int64_t>, std::vector<int64_t>> computeMetrics(
      const std::shared_ptr<fl::Dataset>& pds);

  // convert tokens indices predictions into tokens string
  virtual std::vector<std::string> computeStringPred(
      const std::vector<int>& tokenIdxSeq) = 0;

  // convert tokens indices predictions into tokens string
  virtual std::vector<std::string> computeStringTarget(
      const std::vector<int>& tokenIdxSeq) = 0;

  virtual ~DecodeMaster() = default;

 protected:
  std::shared_ptr<fl::lib::text::Trie> buildTrie(
      const fl::lib::text::LexiconMap& lexicon,
      fl::lib::text::SmearingMode smearMode) const;

  std::shared_ptr<fl::Module> net_;
  std::shared_ptr<fl::lib::text::LM> lm_;
  bool isTokenLM_;
  bool usePlugin_;
  fl::lib::text::Dictionary tokenDict_;
  fl::lib::text::Dictionary wordDict_;
  DecodeMasterTrainOptions trainOpt_;
};

// token-based CTC/ASG decoder (lexicon or lexicon-free)
class TokenDecodeMaster : public DecodeMaster {
 public:
  explicit TokenDecodeMaster(
      const std::shared_ptr<fl::Module> net,
      const std::shared_ptr<fl::lib::text::LM> lm,
      const std::vector<float>& transition,
      const bool usePlugin,
      const fl::lib::text::Dictionary& tokenDict,
      const fl::lib::text::Dictionary& wordDict,
      const DecodeMasterTrainOptions& trainOpt);

  // compute predictions from emissions for lexicon free case
  std::shared_ptr<fl::Dataset> decode(
      const std::shared_ptr<fl::Dataset>& eds,
      DecodeMasterLexiconFreeOptions opt);

  // compute predictions from emissions for lexicon case
  std::shared_ptr<fl::Dataset> decode(
      const std::shared_ptr<fl::Dataset>& eds,
      const fl::lib::text::LexiconMap& lexicon,
      DecodeMasterLexiconOptions opt);

  // convert tokens indices predictions into tokens string
  virtual std::vector<std::string> computeStringPred(
      const std::vector<int>& tokenIdxSeq) override;

  // convert tokens indices predictions into tokens string
  virtual std::vector<std::string> computeStringTarget(
      const std::vector<int>& tokenIdxSeq) override;

 private:
  std::vector<float> transition_;
};

// word-based CTC/ASG decoder (lexicon or lexicon-free)
class WordDecodeMaster : public DecodeMaster {
 public:
  explicit WordDecodeMaster(
      const std::shared_ptr<fl::Module> net,
      const std::shared_ptr<fl::lib::text::LM> lm,
      const std::vector<float>& transition,
      const bool usePlugin,
      const fl::lib::text::Dictionary& tokenDict,
      const fl::lib::text::Dictionary& wordDict,
      const DecodeMasterTrainOptions& trainOpt);

  // compute predictions from emissions
  std::shared_ptr<fl::Dataset> decode(
      const std::shared_ptr<fl::Dataset>& eds,
      const fl::lib::text::LexiconMap& lexicon,
      DecodeMasterLexiconOptions opt);

  // convert tokens indices predictions into tokens string
  virtual std::vector<std::string> computeStringPred(
      const std::vector<int>& tokenIdxSeq) override;

  // convert tokens indices predictions into tokens string
  virtual std::vector<std::string> computeStringTarget(
      const std::vector<int>& tokenIdxSeq) override;

 private:
  std::vector<float> transition_;
};
} // namespace asr
} // namespace app
} // namespace fl
