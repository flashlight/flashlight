/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/libraries/text/tokenizer/Tokenizer.h"

#include <algorithm>
#include <future>

#include "flashlight/libraries/common/String.h"
#include "flashlight/libraries/common/System.h"

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

void Tokenizer::countWords(
    const std::string& filename,
    int numWorkers,
    bool generateDescriptor) {
  // 1. Compute offsets
  auto offsets = findOffsets(filename, numWorkers);

  // 2. Count words in each partition
  std::vector<WordCountMap> subWordCountMaps(numWorkers);
  std::vector<FileDescriptor> subFileDescriptors(numWorkers);
  std::vector<std::future<void>> futures(numWorkers);

  auto countPartialFile = [this](
                              const std::string& filename,
                              size_t begin,
                              size_t end,
                              WordCountMap& wordCountMap,
                              FileDescriptor& fileDescriptor,
                              bool generateDescriptor) {
    auto stream = createInputStream(filename);
    stream.seekg(begin, stream.beg);
    while (stream.tellg() < end) {
      auto words = readAndParseSentence(stream);
      for (const auto& word : words) {
        if (wordCountMap.find(word) == wordCountMap.end()) {
          wordCountMap[word] = 0;
        }
        wordCountMap[word]++;
      }
      if (generateDescriptor) {
        fileDescriptor.push_back(
            {static_cast<size_t>(stream.tellg()), words.size()});
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
        std::ref(subWordCountMaps[i]),
        std::ref(subFileDescriptors[i]),
        generateDescriptor);
  }

  // 3. Merge the counters and file descriptors
  for (int i = 0; i < numWorkers; ++i) {
    futures[i].get();
    // Word counter
    for (const auto& item : subWordCountMaps[i]) {
      if (wordCountMap_.find(item.first) == wordCountMap_.end()) {
        wordCountMap_[item.first] = 0;
      }
      wordCountMap_[item.first] += item.second;
      totalWords_ += item.second;
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

size_t Tokenizer::totalWords() {
  return totalWords_;
}

size_t Tokenizer::totalSentences() {
  return totalSentences_;
}

void Tokenizer::filterWords(int maxWords, int minAppearence) {
  wordCountPairs_.clear();
  for (const auto& item : wordCountMap_) {
    if (item.second > minAppearence) {
      wordCountPairs_.push_back(item);
    }
  }
  std::sort(
      wordCountPairs_.begin(),
      wordCountPairs_.end(),
      [](const WordCountPair& wcp1, const WordCountPair& wcp2) {
        return wcp1.second > wcp2.second;
      });

  maxWords = maxWords > 0 && maxWords < wordCountPairs_.size()
      ? maxWords
      : wordCountPairs_.size();
  wordCountPairs_.resize(maxWords);
}

void Tokenizer::saveDictionary(const std::string& filename) {
  auto stream = createOutputStream(filename);
  for (int i = 0; i < wordCountPairs_.size(); ++i) {
    stream << wordCountPairs_[i].first << " " << wordCountPairs_[i].second
           << "\n";
  }
}

void Tokenizer::saveFileDescriptor(const std::string& filename) {
  auto stream = createOutputStream(filename);
  stream << "words: " << totalWords_ << "\n";
  stream << "sentences: " << totalSentences_ << "\n";
  for (int i = 0; i < fileDescriptor_.size(); ++i) {
    stream << fileDescriptor_[i].first << " " << fileDescriptor_[i].second
           << "\n";
  }
}
} // namespace text
} // namespace lib
} // namespace fl