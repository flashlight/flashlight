/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/decoder/DecodeMaster.h"

#include "flashlight/fl/dataset/MemoryBlobDataset.h"
#include "flashlight/fl/meter/EditDistanceMeter.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/lib/text/decoder/LexiconDecoder.h"
#include "flashlight/lib/text/decoder/LexiconFreeDecoder.h"
#include "flashlight/pkg/runtime/common/SequentialBuilder.h"
#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/decoder/TranscriptionUtils.h"
#include "flashlight/pkg/speech/runtime/Helpers.h"

namespace {

constexpr size_t kDMTokenTargetIdx = 0;
constexpr size_t kDMWordTargetIdx = 1;
constexpr size_t kDMTokenPredIdx = 2;
constexpr size_t kDMWordPredIdx = 3;

using namespace fl;

Tensor removeNegative(const fl::Tensor& arr) {
  return arr(arr >= 0);
}
Tensor removePad(const Tensor& arr, int32_t padIdx) {
  return arr(arr != padIdx);
}
} // namespace

// TODO threading?

namespace fl::pkg::speech {

DecodeMaster::DecodeMaster(
    const std::shared_ptr<fl::Module> net,
    const std::shared_ptr<fl::lib::text::LM> lm,
    const bool isTokenLM,
    const bool usePlugin,
    const fl::lib::text::Dictionary& tokenDict,
    const fl::lib::text::Dictionary& wordDict,
    const DecodeMasterTrainOptions& trainOpt)
    : net_(net),
      lm_(lm),
      isTokenLM_(isTokenLM),
      usePlugin_(usePlugin),
      tokenDict_(tokenDict),
      wordDict_(wordDict),
      trainOpt_(trainOpt) {}

std::pair<std::vector<int64_t>, std::vector<int64_t>>
DecodeMaster::computeMetrics(const std::shared_ptr<fl::Dataset>& predDataset) {
  fl::EditDistanceMeter wordEditDist, tokenEditDist;

  for (auto& sample : *predDataset) {
    if (sample.size() <= kDMWordPredIdx) {
      throw std::runtime_error(
          "computeMetrics: need token/word target to compute WER");
    }
    auto predictionWrd = sample[kDMWordPredIdx];
    auto targetWrd = sample[kDMWordTargetIdx];
    auto prediction = sample[kDMTokenPredIdx];
    auto target = sample[kDMTokenTargetIdx];
    bool isPredictingWrd = !predictionWrd.isEmpty();

    if (prediction.ndim() > 2 || target.ndim() > 2) {
      throw std::runtime_error(
          "computeMetrics: expecting TxB for prediction and target");
    }
    if (isPredictingWrd && (predictionWrd.ndim() > 2 || targetWrd.ndim() > 2)) {
      throw std::runtime_error(
          "computeMetrics: expecting TxB for prediction and target");
    }

    if (!prediction.isEmpty() && !target.isEmpty() &&
        (prediction.dim(1) != target.dim(1))) {
      throw std::runtime_error(
          "computeMetrics: prediction and target do not match");
    }
    if (isPredictingWrd && !predictionWrd.isEmpty() && !targetWrd.isEmpty() &&
        (predictionWrd.dim(1) != targetWrd.dim(1))) {
      throw std::runtime_error(
          "computeMetrics: prediction and target do not match");
    }
    // token predictions and target
    std::vector<int> predictionV = prediction.toHostVector<int>();
    std::vector<int> targetV = target.toHostVector<int>();

    auto predictionS = computeStringPred(predictionV);
    auto targetS = computeStringTarget(targetV);
    tokenEditDist.add(predictionS, targetS);

    std::vector<std::string> targetWrdS, predictionWrdS;
    if (isPredictingWrd) {
      targetWrdS = wrdIdx2Wrd(targetWrd.toHostVector<int>(), wordDict_);
      predictionWrdS = wrdIdx2Wrd(predictionWrd.toHostVector<int>(), wordDict_);
    } else {
      targetWrdS = tkn2Wrd(targetS, trainOpt_.wordSep);
      predictionWrdS = tkn2Wrd(predictionS, trainOpt_.wordSep);
    }
    wordEditDist.add(predictionWrdS, targetWrdS);
  }
  return {tokenEditDist.value(), wordEditDist.value()};
}

std::shared_ptr<fl::lib::text::Trie> DecodeMaster::buildTrie(
    const fl::lib::text::LexiconMap& lexicon,
    fl::lib::text::SmearingMode smearMode) const {
  auto trie = std::make_shared<fl::lib::text::Trie>(
      tokenDict_.indexSize(), tokenDict_.getIndex(trainOpt_.wordSep));
  auto startState = lm_->start(false);
  for (auto& it : lexicon) {
    const std::string& word = it.first;
    int usrIdx = wordDict_.getIndex(word);
    float score = 0;
    if (!isTokenLM_) {
      fl::lib::text::LMStatePtr dummyState;
      std::tie(dummyState, score) = lm_->score(startState, usrIdx);
    }
    for (auto& tokens : it.second) {
      auto tokensTensor = tkn2Idx(tokens, tokenDict_, trainOpt_.repLabel);
      trie->insert(tokensTensor, usrIdx, score);
    }
  }
  // Smearing
  trie->smear(smearMode);
  return trie;
}

std::shared_ptr<fl::Dataset> DecodeMaster::forward(
    const std::shared_ptr<fl::Dataset>& ds) {
  auto emissionDataset = std::make_shared<fl::MemoryBlobDataset>();
  for (auto& batch : *ds) {
    Tensor output;
    if (batch.empty()) {
      continue;
    }
    if (usePlugin_) {
      output = net_->forward({fl::input(batch[kInputIdx]),
                              fl::noGrad(batch[kDurationIdx])})
                   .front()
                   .tensor();
    } else {
      output = fl::pkg::runtime::forwardSequentialModuleWithPadMask(
                   fl::input(batch[kInputIdx]), net_, batch[kDurationIdx])
                   .tensor();
    }
    if (output.ndim() > 3) {
      throw std::runtime_error("output should be NxTxB");
    }
    Tensor tokenTarget =
        (batch.size() > kTargetIdx ? batch[kTargetIdx] : Tensor());
    Tensor wordTarget = (batch.size() > kWordIdx ? batch[kWordIdx] : Tensor());

    int B = output.dim(2);
    if (!tokenTarget.isEmpty() &&
        (tokenTarget.ndim() > 2 || tokenTarget.dim(1) != B)) {
      throw std::runtime_error("token target should be LxB");
    }
    if (!wordTarget.isEmpty() &&
        (wordTarget.ndim() > 2 || wordTarget.dim(1) != B)) {
      throw std::runtime_error("word target should be LxB");
    }
    // todo s2s, if we pad only with -1 we will be good here (not pad with eos)
    for (int b = 0; b < B; b++) {
      std::vector<Tensor> res(4);
      res[kDMTokenPredIdx] = output(fl::span, fl::span, b);
      res[kDMTokenTargetIdx] = removeNegative(tokenTarget(fl::span, b));
      res[kDMTokenTargetIdx] =
          removePad(res[kDMTokenTargetIdx], trainOpt_.targetPadIdx);
      res[kDMWordTargetIdx] = removeNegative(wordTarget(fl::span, b));
      res[kDMWordTargetIdx] =
          removePad(res[kDMWordTargetIdx], trainOpt_.targetPadIdx);
      emissionDataset->add(res);
    }
  }
  emissionDataset->writeIndex();
  return emissionDataset;
}

std::shared_ptr<fl::Dataset> DecodeMaster::decode(
    const std::shared_ptr<fl::Dataset>& emissionDataset,
    fl::lib::text::Decoder& decoder) {
  auto predDataset = std::make_shared<fl::MemoryBlobDataset>();
  for (auto& sample : *emissionDataset) {
    auto emission = sample[kDMTokenPredIdx];
    if (emission.ndim() > 2) {
      throw std::runtime_error("emission should be NxT");
    }
    std::vector<float> emissionV(emission.elements());
    emission.astype(fl::dtype::f32).host(emissionV.data());
    auto results =
        decoder.decode(emissionV.data(), emission.dim(1), emission.dim(0));

    std::vector<int> tokensV, wordsV;
    if (!results.empty()) {
      tokensV = results[0].tokens;
      wordsV = results[0].words;
    }
    tokensV.erase(
        std::remove(tokensV.begin(), tokensV.end(), -1), tokensV.end());
    wordsV.erase(std::remove(wordsV.begin(), wordsV.end(), -1), wordsV.end());
    sample[kDMTokenPredIdx] =
        (!tokensV.empty() ? Tensor::fromVector(tokensV) : Tensor());
    sample[kDMWordPredIdx] =
        (!wordsV.empty() ? Tensor::fromVector(wordsV) : Tensor());
    predDataset->add(sample);
  }
  predDataset->writeIndex();
  return predDataset;
}

TokenDecodeMaster::TokenDecodeMaster(
    const std::shared_ptr<fl::Module> net,
    const std::shared_ptr<fl::lib::text::LM> lm,
    const std::vector<float>& transition,
    const bool usePlugin,
    const fl::lib::text::Dictionary& tokenDict,
    const fl::lib::text::Dictionary& wordDict,
    const DecodeMasterTrainOptions& trainOpt)
    : DecodeMaster(net, lm, true, usePlugin, tokenDict, wordDict, trainOpt),
      transition_(transition) {}

std::shared_ptr<fl::Dataset> TokenDecodeMaster::decode(
    const std::shared_ptr<fl::Dataset>& emissionDataset,
    DecodeMasterLexiconFreeOptions opt) {
  fl::lib::text::LexiconFreeDecoderOptions decoderOpt{
      .beamSize = opt.beamSize,
      .beamSizeToken = opt.beamSizeToken,
      .beamThreshold = opt.beamThreshold,
      .lmWeight = opt.lmWeight,
      .silScore = opt.silScore,
      .logAdd = opt.logAdd,
      .criterionType = fl::lib::text::CriterionType::CTC};
  auto silIdx = tokenDict_.getIndex(opt.silToken);
  auto blankIdx = tokenDict_.getIndex(opt.blankToken);
  fl::lib::text::LexiconFreeDecoder decoder(
      decoderOpt, lm_, silIdx, blankIdx, transition_);
  return DecodeMaster::decode(emissionDataset, decoder);
}

std::shared_ptr<fl::Dataset> TokenDecodeMaster::decode(
    const std::shared_ptr<fl::Dataset>& emissionDataset,
    const fl::lib::text::LexiconMap& lexicon,
    DecodeMasterLexiconOptions opt) {
  auto trie = buildTrie(lexicon, opt.smearMode);
  fl::lib::text::LexiconDecoderOptions decoderOpt{
      .beamSize = opt.beamSize,
      .beamSizeToken = opt.beamSizeToken,
      .beamThreshold = opt.beamThreshold,
      .lmWeight = opt.lmWeight,
      .wordScore = opt.wordScore,
      .unkScore = opt.unkScore,
      .silScore = opt.silScore,
      .logAdd = opt.logAdd,
      .criterionType = fl::lib::text::CriterionType::CTC};
  auto silIdx = tokenDict_.getIndex(opt.silToken);
  auto blankIdx = tokenDict_.getIndex(opt.blankToken);
  auto unkWordIdx = wordDict_.getIndex(fl::lib::text::kUnkToken);
  fl::lib::text::LexiconDecoder decoder(
      decoderOpt, trie, lm_, silIdx, blankIdx, unkWordIdx, transition_, true);
  return DecodeMaster::decode(emissionDataset, decoder);
}

std::vector<std::string> TokenDecodeMaster::computeStringPred(
    const std::vector<int>& tokenIdxSeq) {
  return tknPrediction2Ltr(
      tokenIdxSeq,
      tokenDict_,
      "ctc",
      trainOpt_.surround,
      false, // eosToken
      trainOpt_.repLabel,
      trainOpt_.wordSepIsPartOfToken,
      trainOpt_.wordSep);
}

std::vector<std::string> TokenDecodeMaster::computeStringTarget(
    const std::vector<int>& tokenIdxSeq) {
  return tknTarget2Ltr(
      tokenIdxSeq,
      tokenDict_,
      "ctc",
      trainOpt_.surround,
      false, // eosToken
      trainOpt_.repLabel,
      trainOpt_.wordSepIsPartOfToken,
      trainOpt_.wordSep);
}

WordDecodeMaster::WordDecodeMaster(
    const std::shared_ptr<fl::Module> net,
    const std::shared_ptr<fl::lib::text::LM> lm,
    const std::vector<float>& transition,
    const bool usePlugin,
    const fl::lib::text::Dictionary& tokenDict,
    const fl::lib::text::Dictionary& wordDict,
    const DecodeMasterTrainOptions& trainOpt)
    : DecodeMaster(net, lm, false, usePlugin, tokenDict, wordDict, trainOpt),
      transition_(transition) {}

std::shared_ptr<fl::Dataset> WordDecodeMaster::decode(
    const std::shared_ptr<fl::Dataset>& emissionDataset,
    const fl::lib::text::LexiconMap& lexicon,
    DecodeMasterLexiconOptions opt) {
  auto trie = buildTrie(lexicon, opt.smearMode);
  fl::lib::text::LexiconDecoderOptions decoderOpt{
      .beamSize = opt.beamSize,
      .beamSizeToken = opt.beamSizeToken,
      .beamThreshold = opt.beamThreshold,
      .lmWeight = opt.lmWeight,
      .wordScore = opt.wordScore,
      .unkScore = opt.unkScore,
      .silScore = opt.silScore,
      .logAdd = opt.logAdd,
      .criterionType = fl::lib::text::CriterionType::CTC};
  auto silIdx = tokenDict_.getIndex(opt.silToken);
  auto blankIdx = tokenDict_.getIndex(opt.blankToken);
  auto unkWordIdx = wordDict_.getIndex(opt.unkToken);
  fl::lib::text::LexiconDecoder decoder(
      decoderOpt, trie, lm_, silIdx, blankIdx, unkWordIdx, transition_, false);
  return DecodeMaster::decode(emissionDataset, decoder);
}

std::vector<std::string> WordDecodeMaster::computeStringPred(
    const std::vector<int>& tokenIdxSeq) {
  return tknPrediction2Ltr(
      tokenIdxSeq,
      tokenDict_,
      "ctc",
      trainOpt_.surround,
      false, // eosToken
      trainOpt_.repLabel,
      trainOpt_.wordSepIsPartOfToken,
      trainOpt_.wordSep);
}

std::vector<std::string> WordDecodeMaster::computeStringTarget(
    const std::vector<int>& tokenIdxSeq) {
  return tknTarget2Ltr(
      tokenIdxSeq,
      tokenDict_,
      "ctc",
      trainOpt_.surround,
      false, // eosToken
      trainOpt_.repLabel,
      trainOpt_.wordSepIsPartOfToken,
      trainOpt_.wordSep);
}

} // namespace fl
