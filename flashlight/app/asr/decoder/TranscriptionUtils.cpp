/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/decoder/TranscriptionUtils.h"

#include "flashlight/lib/common/String.h"

using namespace fl::lib;
using fl::lib::text::Dictionary;
using fl::lib::text::splitWrd;

namespace fl {
namespace app {
namespace asr {

std::vector<std::string> tknIdx2Ltr(
    const std::vector<int>& labels,
    const Dictionary& d,
    const bool useWordPiece,
    const std::string& wordSep) {
  std::vector<std::string> result;

  for (auto id : labels) {
    auto token = d.getEntry(id);
    if (useWordPiece) {
      auto splitToken = splitWrd(token);
      for (const auto& c : splitToken) {
        result.emplace_back(c);
      }
    } else {
      result.emplace_back(token);
    }
  }

  if (result.size() > 0 && !wordSep.empty()) {
    if (result.front() == wordSep) {
      result.erase(result.begin());
    }
    if (!result.empty() && result.back() == wordSep) {
      result.pop_back();
    }
  }

  return result;
}

std::vector<std::string> tkn2Wrd(
    const std::vector<std::string>& input,
    const std::string& wordSep) {
  std::vector<std::string> words;
  std::string currentWord = "";
  for (auto& tkn : input) {
    if (tkn == wordSep) {
      if (!currentWord.empty()) {
        words.push_back(currentWord);
        currentWord = "";
      }
    } else {
      currentWord += tkn;
    }
  }
  if (!currentWord.empty()) {
    words.push_back(currentWord);
  }
  return words;
}

std::vector<std::string> wrdIdx2Wrd(
    const std::vector<int>& input,
    const Dictionary& wordDict) {
  std::vector<std::string> words;
  for (auto wrdIdx : input) {
    words.push_back(wordDict.getEntry(wrdIdx));
  }
  return words;
}

std::vector<std::string> tknTarget2Ltr(
    std::vector<int> tokens,
    const Dictionary& tokenDict,
    const std::string& criterion,
    const std::string& surround,
    const bool eosTkn,
    const int replabel,
    const bool useWordPiece,
    const std::string& wordSep) {
  if (tokens.empty()) {
    return std::vector<std::string>{};
  }

  if (criterion == kSeq2SeqCriterion) {
    if (tokens.back() == tokenDict.getIndex(kEosToken)) {
      tokens.pop_back();
    }
  }
  remapLabels(tokens, tokenDict, surround, eosTkn, replabel);

  return tknIdx2Ltr(tokens, tokenDict, useWordPiece, wordSep);
}

std::vector<std::string> tknPrediction2Ltr(
    std::vector<int> tokens,
    const Dictionary& tokenDict,
    const std::string& criterion,
    const std::string& surround,
    const bool eosTkn,
    const int replabel,
    const bool useWordPiece,
    const std::string& wordSep) {
  if (tokens.empty()) {
    return std::vector<std::string>{};
  }

  if (criterion == kCtcCriterion || criterion == kAsgCriterion) {
    dedup(tokens);
  }
  if (criterion == kCtcCriterion) {
    int blankIdx = tokenDict.getIndex(kBlankToken);
    tokens.erase(
        std::remove(tokens.begin(), tokens.end(), blankIdx), tokens.end());
  }
  tokens = validateIdx(tokens, -1);
  remapLabels(tokens, tokenDict, surround, eosTkn, replabel);

  return tknIdx2Ltr(tokens, tokenDict, useWordPiece, wordSep);
}

std::vector<int> tkn2Idx(
    const std::vector<std::string>& spelling,
    const Dictionary& tokenDict,
    int maxReps) {
  std::vector<int> ret;
  ret.reserve(spelling.size());
  for (const auto& token : spelling) {
    ret.push_back(tokenDict.getIndex(token));
  }
  return packReplabels(ret, tokenDict, maxReps);
}

std::vector<int> validateIdx(std::vector<int> input, int unkIdx) {
  int newSize = 0;
  for (int i = 0; i < input.size(); i++) {
    if (input[i] >= 0 and input[i] != unkIdx) {
      input[newSize] = input[i];
      newSize++;
    }
  }
  input.resize(newSize);

  return input;
}
} // namespace asr
} // namespace app
} // namespace fl
