/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Utils.h"

#include <iostream>

#include "common/Defines.h"

using namespace fl::lib;

namespace fl {
namespace task {
namespace asr {

std::vector<std::string> wrd2Target(
    const std::string& word,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    bool fallback2Ltr /* = false */,
    bool skipUnk /* = false */) {
  auto lit = lexicon.find(word);
  if (lit != lexicon.end()) {
    if (lit->second.size() > 1 &&
        FLAGS_sampletarget >
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) {
      return lit->second[std::rand() % lit->second.size()];
    } else {
      return lit->second[0];
    }
  }

  std::vector<std::string> res;
  if (fallback2Ltr) {
    std::cerr
        << "Falling back to using letters as targets for the unknown word: "
        << word << "\n";
    auto tokens = splitWrd(word);
    for (const auto& tkn : tokens) {
      if (dict.contains(tkn)) {
        res.push_back(tkn);
      } else if (skipUnk) {
        std::cerr
            << "Skipping unknown token '" << tkn
            << "' when falling back to letter target for the unknown word: "
            << word << "\n";
      } else {
        throw std::invalid_argument(
            "Unknown token '" + tkn +
            "' when falling back to letter target for the unknown word: " +
            word);
      }
    }
  } else if (skipUnk) {
    std::cerr << "Skipping unknown word '" << word
              << "' when generating target\n";
  } else {
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
  std::vector<std::string> res;
  for (auto w : words) {
    auto t = wrd2Target(w, lexicon, dict, fallback2Ltr, skipUnk);

    if (t.size() == 0) {
      continue;
    }

    // remove duplicate word separators in the beginning of each target token
    if (res.size() > 0 && !FLAGS_wordseparator.empty() &&
        t[0].length() >= FLAGS_wordseparator.length() &&
        t[0].compare(0, FLAGS_wordseparator.length(), FLAGS_wordseparator) ==
            0) {
      res.pop_back();
    }

    res.insert(res.end(), t.begin(), t.end());

    if (!FLAGS_wordseparator.empty() &&
        !(res.back().length() >= FLAGS_wordseparator.length() &&
          res.back().compare(
              res.back().length() - FLAGS_wordseparator.length(),
              FLAGS_wordseparator.length(),
              FLAGS_wordseparator) == 0)) {
      res.emplace_back(FLAGS_wordseparator);
    }
  }

  if (res.size() > 0 && res.back() == FLAGS_wordseparator) {
    res.pop_back();
  }
  return res;
}

} 
}
}
