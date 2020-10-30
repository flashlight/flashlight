/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/lib/text/tokenizer/Tokenizer.h"

#include <algorithm>
#include <future>

#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

namespace fl {
namespace lib {
namespace text {

std::vector<size_t> Tokenizer::findOffsets(
    const std::string& filename,
    int nWorkers) {
  auto stream = createInputStream(filename);
  stream.seekg(0, stream.end);
  size_t fileSize = stream.tellg();
  stream.seekg(0, stream.beg);

  size_t chunkSize = fileSize / nWorkers;
  std::vector<size_t> offsets(nWorkers + 1, 0);

  std::string line;
  for (int i = 1; i < nWorkers; ++i) {
    stream.seekg(chunkSize * i, std::ios::beg);
    // TODO: be careful of corner cases
    std::getline(stream, line);
    offsets[i] = stream.tellg();
  }
  offsets.back() = fileSize;

  stream.close();
  return offsets;
}

std::vector<std::string> Tokenizer::readAndParseSentence(
    std::ifstream& stream) {
  std::string line;
  std::getline(stream, line);
  return splitOnWhitespace(line, true);
}

void Tokenizer::countTokens(
    const std::string& filename,
    int numWorkers,
    bool generateDescriptor) {
  // 1. Compute offsets
  auto offsets = findOffsets(filename, numWorkers);

  // 2. Count tokens in each partition
  std::vector<TokenCountMap> subTokenCountMaps(numWorkers);
  std::vector<FileDescriptor> subFileDescriptors(numWorkers);
  std::vector<std::future<void>> futures(numWorkers);

  auto countPartialFile = [this](
                              const std::string& filename,
                              size_t begin,
                              size_t end,
                              TokenCountMap& tokenCountMap,
                              FileDescriptor& fileDescriptor,
                              bool generateDescriptor) {
    auto stream = createInputStream(filename);
    stream.seekg(begin, stream.beg);
    while (stream.tellg() < end) {
      auto tokens = readAndParseSentence(stream);
      for (const auto& token : tokens) {
        if (tokenCountMap.find(token) == tokenCountMap.end()) {
          tokenCountMap[token] = 0;
        }
        tokenCountMap[token]++;
      }
      if (generateDescriptor) {
        fileDescriptor.push_back(
            {static_cast<size_t>(stream.tellg()), tokens.size()});
      }
    }
    stream.close();
  };

  for (int i = 0; i < numWorkers; ++i) {
    futures[i] = std::async(
        std::launch::async,
        countPartialFile,
        filename,
        offsets[i],
        offsets[i + 1],
        std::ref(subTokenCountMaps[i]),
        std::ref(subFileDescriptors[i]),
        generateDescriptor);
  }

  // 3. Merge the counters and file descriptors
  for (int i = 0; i < numWorkers; ++i) {
    futures[i].get();
    // Token counter
    for (const auto& item : subTokenCountMaps[i]) {
      if (tokenCountMap_.find(item.first) == tokenCountMap_.end()) {
        tokenCountMap_[item.first] = 0;
      }
      tokenCountMap_[item.first] += item.second;
      totalTokens_ += item.second;
    }
    // File descriptors
    if (generateDescriptor) {
      fileDescriptor_.insert(
          fileDescriptor_.end(),
          subFileDescriptors[i].begin(),
          subFileDescriptors[i].end());
      totalSentences_ += fileDescriptor_.size();
    }
  }
}

size_t Tokenizer::totalTokens() {
  return totalTokens_;
}

size_t Tokenizer::totalSentences() {
  return totalSentences_;
}

void Tokenizer::filterTokens(int maxTokens, int minAppearence) {
  tokenCountPairs_.clear();
  for (const auto& item : tokenCountMap_) {
    if (item.second > minAppearence) {
      tokenCountPairs_.push_back(item);
    }
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

  maxTokens = maxTokens > -1 && maxTokens < tokenCountPairs_.size()
      ? maxTokens
      : tokenCountPairs_.size();
  tokenCountPairs_.resize(maxTokens);
}

void Tokenizer::saveDictionary(const std::string& filename) {
  auto stream = createOutputStream(filename);
  for (int i = 0; i < tokenCountPairs_.size(); ++i) {
    stream << tokenCountPairs_[i].first << " " << tokenCountPairs_[i].second
           << "\n";
  }
}

void Tokenizer::saveFileDescriptor(const std::string& filename) {
  auto stream = createOutputStream(filename);
  stream << "tokens: " << totalTokens_ << "\n";
  stream << "sentences: " << totalSentences_ << "\n";
  for (int i = 0; i < fileDescriptor_.size(); ++i) {
    stream << fileDescriptor_[i].first << " " << fileDescriptor_[i].second
           << "\n";
  }
}
} // namespace text
} // namespace lib
} // namespace fl
