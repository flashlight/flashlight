/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/data/Utils.h"

#include <iostream>

#include "flashlight/app/asr/common/Defines.h"

using fl::lib::text::Dictionary;
using fl::lib::text::LexiconMap;
using fl::lib::text::splitWrd;

namespace fl {
namespace app {
namespace asr {

std::vector<std::string> wrd2Target(
    const std::string& word,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    bool fallback2Ltr /* = false */,
    bool skipUnk /* = false */) {
  bool fallback2LtrWordSepLeft = fallback2Ltr && FLAGS_usewordpiece;
  bool fallback2LtrWordSepRight = fallback2Ltr && !FLAGS_usewordpiece;
  return wrd2Target(
      word, 
      lexicon, 
      dict, 
      FLAGS_wordseparator,
      FLAGS_sampletarget, 
      fallback2LtrWordSepLeft, 
      fallback2LtrWordSepRight, 
      skipUnk);
}

std::vector<std::string> wrd2Target(
    const std::string& word,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    const std::string& wordSeparator /* = "" */,
    float targetSamplePct /* = 0 */,
    bool fallback2LtrWordSepLeft /* = false */,
    bool fallback2LtrWordSepRight /* = false */,
    bool skipUnk /* = false */) {
  // find the word in the lexicon and use its spelling
  auto lit = lexicon.find(word);
  if (lit != lexicon.end()) {
    // sample random spelling if word has different spellings
    if (lit->second.size() > 1 &&
        targetSamplePct >
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) {
      return lit->second[std::rand() % lit->second.size()];
    } else {
      return lit->second[0];
    }
  } 
  
  std::vector<std::string> word2tokens;
  if (fallback2LtrWordSepLeft || fallback2LtrWordSepRight) {
    if (fallback2LtrWordSepLeft && !wordSeparator.empty()) {
      // add word separator at the beginning of fallback word 
      word2tokens.push_back(wordSeparator);
    } 
    auto tokens = splitWrd(word);
    for (const auto& tkn : tokens) {
      if (dict.contains(tkn)) {
        word2tokens.push_back(tkn);
      } else if (!skipUnk) {
        throw std::invalid_argument(
            "Unknown token '" + tkn +
            "' when falling back to letter target for the unknown word: " +
            word);
      }
    }
    if (fallback2LtrWordSepRight && !wordSeparator.empty()) {
      // add word separator at the end of fallback word 
      word2tokens.push_back(wordSeparator);
    }
  } else if (!skipUnk) {
    throw std::invalid_argument("Unknown word in the lexicon: " + word);
  }
  return word2tokens;
}

std::vector<std::string> wrd2Target(
    const std::vector<std::string>& words,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    bool fallback2Ltr /* = false */,
    bool skipUnk /* = false */) {
  bool fallback2LtrWordSepLeft = fallback2Ltr && FLAGS_usewordpiece;
  bool fallback2LtrWordSepRight = fallback2Ltr && !FLAGS_usewordpiece;
  return wrd2Target(
      words,
      lexicon,
      dict,
      FLAGS_wordseparator,
      FLAGS_sampletarget,
      fallback2LtrWordSepLeft,
      fallback2LtrWordSepRight,
      skipUnk);
}

std::vector<std::string> wrd2Target(
    const std::vector<std::string>& words,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    const std::string& wordSeparator /* = "" */,
    float targetSamplePct /* = 0 */,
    bool fallback2LtrWordSepLeft /* = false */,
    bool fallback2LtrWordSepRight /* = false */,
    bool skipUnk /* = false */) {
  std::vector<std::string> res;
  for (auto w : words) {
    auto w2tokens =
        wrd2Target(
          w, 
          lexicon, 
          dict, 
          wordSeparator,
          targetSamplePct, 
          fallback2LtrWordSepLeft, 
          fallback2LtrWordSepRight, 
          skipUnk);

    if (w2tokens.size() == 0) {
      continue;
    }
    res.insert(res.end(), w2tokens.begin(), w2tokens.end());
  }
  return res;
}
} // namespace asr
} // namespace app
} // namespace fl
