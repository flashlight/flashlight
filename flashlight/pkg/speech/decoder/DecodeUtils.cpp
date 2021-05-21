/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/decoder/DecodeUtils.h"
using fl::lib::text::SmearingMode;

namespace fl {
namespace app {
namespace asr {

std::shared_ptr<fl::lib::text::Trie> buildTrie(
    const std::string& decoderType,
    bool useLexicon,
    std::shared_ptr<fl::lib::text::LM> lm,
    const std::string& smearing,
    const fl::lib::text::Dictionary& tokenDict,
    const fl::lib::text::LexiconMap& lexicon,
    const fl::lib::text::Dictionary& wordDict,
    const int wordSeparatorIdx,
    const int repLabel) {
  if (!(decoderType == "wrd" || useLexicon)) {
    return nullptr;
  }
  auto trie = std::make_shared<fl::lib::text::Trie>(
      tokenDict.indexSize(), wordSeparatorIdx);
  auto startState = lm->start(false);

  for (auto& it : lexicon) {
    const std::string& word = it.first;
    int usrIdx = wordDict.getIndex(word);
    float score = -1;
    if (decoderType == "wrd") {
      fl::lib::text::LMStatePtr dummyState;
      std::tie(dummyState, score) = lm->score(startState, usrIdx);
    }
    for (auto& tokens : it.second) {
      auto tokensTensor = tkn2Idx(tokens, tokenDict, repLabel);
      trie->insert(tokensTensor, usrIdx, score);
    }
  }
  // Smearing
  SmearingMode smearMode = SmearingMode::NONE;
  if (smearing == "logadd") {
    smearMode = SmearingMode::LOGADD;
  } else if (smearing == "max") {
    smearMode = SmearingMode::MAX;
  } else if (smearing != "none") {
    throw std::runtime_error(
        "[buildTrie] Invalid smearing option, can be {logadd, max, none}, provided value is " +
        smearing);
  }
  trie->smear(smearMode);
  return trie;
}

} // namespace asr
} // namespace app
} // namespace fl
