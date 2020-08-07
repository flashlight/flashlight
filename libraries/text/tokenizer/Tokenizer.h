/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace fl {
namespace lib {
namespace text {

class Tokenizer {
  using WordCountMap = std::unordered_map<std::string, size_t>;
  using WordCountPair = std::pair<std::string, size_t>;
  using FileDescriptor = std::vector<std::pair<size_t, int>>;

 public:
  Tokenizer() {}

  std::vector<size_t> findOffsets(const std::string& filename, int nWorkers);

  std::vector<std::string> readAndParseSentence(std::ifstream& stream);

  void countWords(
      const std::string& filename,
      int numWorkers = 1,
      bool generateDescriptor = false);

  void filterWords(int maxWords = 0, int minAppearence = 0);

  void saveDictionary(const std::string& filename);

  void saveFileDescriptor(const std::string& filename);

  size_t totalWords();

  size_t totalSentences();

 private:
  size_t totalWords_{0};
  size_t totalSentences_{0};

  WordCountMap wordCountMap_;
  FileDescriptor fileDescriptor_;
  std::vector<WordCountPair> wordCountPairs_;
};
} // namespace text
} // namespace lib
} // namespace fl