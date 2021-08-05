/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace fl {
namespace lib {
namespace text {

// TextFileMetaData saves the meta data info of each sentence
// 1) the offset it begins from in the text file.
// 2) the number of tokens it contains.
using TextFileMetaData = std::vector<std::pair<size_t, int>>;
using TokenCountPair = std::pair<std::string, size_t>;

/**
 * Tokenizer is designed to tokenize a given chunk of text.
 * It also supports to compute statistics of words in a given text dataset as
 * well as generate meta data for the raw text files.
 *
 * Usage:
 *
 * Tokenizer tokenizer();
 * for (textFile in dataset) {
 *   tokenizer.countTokens(textFile, nWorkers, true);
 *   auto fileMetaData = tokenizer.getTextFileMetaData();
 *   // Do something with fileMetaData
 * }
 *
 * tokenizer.pruneTokens(maxSize, minAppearence);
 * auto tokenCountPairs = tokenizer.getDictionary();
 * // Do something with the tokens
 *
 * -------------------------------
 *
 * This is still an early implementation, which only supports:
 * - Tokenizing space-separated text
 * - None unicode encoded text
 *
 * TODO:
 * - Support different word separator.
 * - Support multilingal/unicode use cases.
 */
class Tokenizer {
 public:
  Tokenizer() {}

  std::vector<std::string> tokenize(const std::string& sentence) const;

  void countTokens(
      const std::string& filename,
      int numWorkers = 1,
      bool generateMetaData = false);
  void pruneTokens(int maxTokens = -1, int minAppearence = 0);

  std::vector<TokenCountPair> getDictionary() const;
  TextFileMetaData getTextFileMetaData() const;

  size_t totalTokens() const;
  size_t totalSentences() const;

 private:
  size_t totalTokens_{0};
  size_t totalSentences_{0};

  TextFileMetaData fileMetaData_;
  std::vector<TokenCountPair> tokenCountPairs_;
};

} // namespace text
} // namespace lib
} // namespace fl
