/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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
  using TokenCountMap = std::unordered_map<std::string, size_t>;
  using TokenCountPair = std::pair<std::string, size_t>;
  using FileDescriptor = std::vector<std::pair<size_t, int>>;

 public:
  Tokenizer() {}

  std::vector<size_t> findOffsets(const std::string& filename, int nWorkers);

  std::vector<std::string> readAndParseSentence(std::ifstream& stream);

  void countTokens(
      const std::string& filename,
      int numWorkers = 1,
      bool generateDescriptor = false);

  void filterTokens(int maxTokens = -1, int minAppearence = 0);

  void saveDictionary(const std::string& filename);

  void saveFileDescriptor(const std::string& filename);

  size_t totalTokens();

  size_t totalSentences();

 private:
  size_t totalTokens_{0};
  size_t totalSentences_{0};

  TokenCountMap tokenCountMap_;
  FileDescriptor fileDescriptor_;
  std::vector<TokenCountPair> tokenCountPairs_;
};
} // namespace text
} // namespace lib
} // namespace fl
