/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
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
  return wrd2Target(
      word, lexicon, dict, FLAGS_sampletarget, fallback2Ltr, skipUnk);
}

std::vector<std::string> wrd2Target(
    const std::string& word,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    float targetSamplePct /* = 0 */,
    bool fallback2Ltr /* = false */,
    bool skipUnk /* = false */) {
  auto lit = lexicon.find(word);
  if (lit != lexicon.end()) {
    if (lit->second.size() > 1 &&
        targetSamplePct >
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) {
      return lit->second[std::rand() % lit->second.size()];
    } else {
      return lit->second[0];
    }
  }

  std::vector<std::string> res;
  if (fallback2Ltr) {
    auto tokens = splitWrd(word);
    for (const auto& tkn : tokens) {
      if (dict.contains(tkn)) {
        res.push_back(tkn);
      } else if (!skipUnk) {
        throw std::invalid_argument(
            "Unknown token '" + tkn +
            "' when falling back to letter target for the unknown word: " +
            word);
      }
    }
  } else if (!skipUnk) {
    throw std::invalid_argument("Unknown word in the lexicon: " + word);
  }
  return res;
}

std::vector<std::string> wrd2Target(
    const std::vector<std::string>& words,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    bool fallback2Ltr /* = false */,
    bool skipUnk /* = false */) {
  return wrd2Target(
      words,
      lexicon,
      dict,
      FLAGS_wordseparator,
      FLAGS_sampletarget,
      fallback2Ltr,
      skipUnk);
}

std::vector<std::string> wrd2Target(
    const std::vector<std::string>& words,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    const std::string& wordSeparator /* = "" */,
    float targetSamplePct /* = 0 */,
    bool fallback2Ltr /* = false */,
    bool skipUnk /* = false */) {
  std::vector<std::string> res;
  for (auto w : words) {
    auto t =
        wrd2Target(w, lexicon, dict, targetSamplePct, fallback2Ltr, skipUnk);

    if (t.size() == 0) {
      continue;
    }

    // remove duplicate word separators in the beginning of each target token
    if (res.size() > 0 && !wordSeparator.empty() &&
        t[0].length() >= wordSeparator.length() &&
        t[0].compare(0, wordSeparator.length(), wordSeparator) == 0) {
      res.pop_back();
    }

    res.insert(res.end(), t.begin(), t.end());

    if (!wordSeparator.empty() &&
        !(res.back().length() >= wordSeparator.length() &&
          res.back().compare(
              res.back().length() - wordSeparator.length(),
              wordSeparator.length(),
              wordSeparator) == 0)) {
      res.emplace_back(wordSeparator);
    }
  }

  if (res.size() > 0 && res.back() == wordSeparator) {
    res.pop_back();
  }
  return res;
}
} // namespace asr
} // namespace app
} // namespace fl
