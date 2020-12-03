/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/lib/text/tokenizer/Tokenizer.h"

#include <algorithm>
#include <future>

#include "flashlight/lib/text/tokenizer/PartialFileReader.h"

namespace fl {
namespace lib {
namespace text {

using TokenCountMap = std::unordered_map<std::string, size_t>;

std::vector<std::string> Tokenizer::tokenize(
    const std::string& sentence) const {
  return splitOnWhitespace(sentence, true);
}

void Tokenizer::countTokens(
    const std::string& filename,
    int numWorkers,
    bool generateMetaData) {
  std::vector<TokenCountMap> subTokenCountMaps(numWorkers);
  std::vector<TextFileMetaData> subTextFileMetaDatas(numWorkers);
  std::vector<std::future<int>> futures(numWorkers);

  auto countPartialFile = [this, numWorkers](
                              const std::string& filename,
                              int rank,
                              TokenCountMap& tokenCountMap,
                              TextFileMetaData& fileMetaData,
                              bool generateMetaData) -> int {
    PartialFileReader reader(rank, numWorkers);
    reader.loadFile(filename);
    int nSentences = 0;
    while (reader.hasNextLine()) {
      auto tokens = tokenize(reader.getLine());
      for (const auto& token : tokens) {
        if (tokenCountMap.find(token) == tokenCountMap.end()) {
          tokenCountMap[token] = 0;
        }
        tokenCountMap[token]++;
      }
      if (generateMetaData) {
        fileMetaData.push_back({reader.getPosition(), tokens.size()});
      }
      nSentences++;
    }

    return nSentences;
  };

  /* 1. Launch threads */
  for (int i = 0; i < numWorkers; ++i) {
    futures[i] = std::async(
        std::launch::async,
        countPartialFile,
        filename,
        i,
        std::ref(subTokenCountMaps[i]),
        std::ref(subTextFileMetaDatas[i]),
        generateMetaData);
  }

  /* 2. Gather results */
  fileMetaData_.clear();
  TokenCountMap tokenCountMap;
  for (int i = 0; i < numWorkers; ++i) {
    totalSentences_ += futures[i].get();
    // Token counter
    for (const auto& item : subTokenCountMaps[i]) {
      if (tokenCountMap.find(item.first) == tokenCountMap.end()) {
        tokenCountMap[item.first] = 0;
      }
      tokenCountMap[item.first] += item.second;
      totalTokens_ += item.second;
    }
    // File MetaDatas
    if (generateMetaData) {
      fileMetaData_.insert(
          fileMetaData_.end(),
          subTextFileMetaDatas[i].begin(),
          subTextFileMetaDatas[i].end());
    }
  }

  /* 3. Sort tokens */
  tokenCountPairs_.clear();
  for (const auto& item : tokenCountMap) {
    tokenCountPairs_.push_back(item);
  }
  std::sort(
      tokenCountPairs_.begin(),
      tokenCountPairs_.end(),
      [](const TokenCountPair& wcp1, const TokenCountPair& wcp2) {
        if (wcp1.second != wcp2.second) {
          // sort by occurences
          return wcp1.second > wcp2.second;
        } else {
          // sort in lexical order if occurences is the same
          return wcp1.first < wcp2.first;
        }
      });
}

void Tokenizer::pruneTokens(int maxTokens, int minAppearence) {
  maxTokens = maxTokens > -1 && maxTokens < tokenCountPairs_.size()
      ? maxTokens
      : tokenCountPairs_.size();
  tokenCountPairs_.resize(maxTokens);

  auto end = std::find_if_not(
      tokenCountPairs_.begin(),
      tokenCountPairs_.end(),
      [&minAppearence](const TokenCountPair& pair) {
        return pair.second >= minAppearence;
      });
  tokenCountPairs_.resize(std::distance(tokenCountPairs_.begin(), end));
}

std::vector<TokenCountPair> Tokenizer::getDictionary() const {
  return tokenCountPairs_;
}

TextFileMetaData Tokenizer::getTextFileMetaData() const {
  return fileMetaData_;
}

size_t Tokenizer::totalTokens() const {
  return totalTokens_;
}

size_t Tokenizer::totalSentences() const {
  return totalSentences_;
}

} // namespace text
} // namespace lib
} // namespace fl
