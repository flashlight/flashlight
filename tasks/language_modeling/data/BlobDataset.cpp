/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "BlobDataset.h"

#include "libraries/common/String.h"
#include "libraries/common/System.h"

using namespace fl::lib;

namespace fl {
namespace task {
namespace lm {

BlobDataset::BlobDataset(
    const std::string& dataDirectory,
    const std::string& filenames,
    const Dictionary& dictionary,
    int worldRank,
    int worldSize,
    int64_t tokensPerSample,
    int64_t batchSize,
    int pad,
    int eos,
    const std::string& sampleBreakMode /* = "none" */,
    bool useDynamicBatching /* = false */)
    : dictionary_(dictionary), tokenizer_(), pad_(pad) {
  data_.clear();
  data_.push_back(eos); // prepend eos

  // 1. Read data
  std::vector<std::pair<int64_t, int64_t>> sentenceEnds;
  auto files = lib::split(',', filenames);
  for (const auto& file : files) {
    auto path = pathsConcat(dataDirectory, file);
    auto offsets = tokenizer_.findOffsets(path, worldSize);
    auto rangeStart = offsets[worldRank];
    auto rangeEnd = offsets[worldRank + 1];

    auto stream = createInputStream(path);
    stream.seekg(rangeStart, stream.beg);
    while (stream.tellg() < rangeEnd) {
      auto words = tokenizer_.readAndParseSentence(stream);
      auto indices = dictionary_.mapEntriesToIndices(words);
      data_.insert(data_.end(), indices.begin(), indices.end());
      data_.push_back(eos);
    }
    stream.close();
  }
  sentenceEnds.pop_back();
  int64_t numTokens = data_.size();

  // 2. Form batches
  if (sampleBreakMode == "none") {
    // Block raw data into samples of size `tokensPerSample`.
    int64_t numSamples = (numTokens + tokensPerSample - 1) / tokensPerSample;
    int64_t numBatches = (numSamples + batchSize - 1) / batchSize;
    for (int64_t b = 0; b < numBatches; ++b) {
      int64_t firstSample = b * batchSize;
      int64_t lastSample = std::min((b + 1) * batchSize, numSamples);
      std::vector<SamplePosition> batch;
      for (int64_t s = firstSample; s < lastSample; ++s) {
        int64_t firstToken = s * tokensPerSample;
        int64_t lastToken = std::min((s + 1) * tokensPerSample, numTokens);
        batch.push_back(SamplePosition{firstToken, lastToken});
      }
      batches_.push_back(std::move(batch));
    }
  } else if (sampleBreakMode == "eos") {
    // Block raw data into samples of completed sentences.
    std::sort(
        sentenceEnds.begin(),
        sentenceEnds.end(),
        [](const std::pair<int64_t, int64_t>& p1,
           const std::pair<int64_t, int64_t>& p2) {
          return p1.second - p1.first < p2.second - p2.first;
        });

    std::vector<SamplePosition> batch;
    for (int64_t i = 0; i < sentenceEnds.size(); ++i) {
      const auto& startPoint = sentenceEnds[i].first;
      const auto& endPoint = sentenceEnds[i].second;
      const auto sampleSize = endPoint - startPoint + 1;
      if (sampleSize > tokensPerSample) {
        break;
      }
      batch.push_back(SamplePosition{startPoint - 1, endPoint - 1});
      if (!useDynamicBatching) {
        if (batch.size() == batchSize) {
          batches_.push_back(std::move(batch));
          batch.clear();
        }
      } else {
        if ((batch.size() + 1) * sampleSize > tokensPerSample) {
          batches_.push_back(std::move(batch));
          batch.clear();
        }
      }
    }
    if (!batch.empty()) {
      batches_.push_back(std::move(batch));
    }
    shuffle(0);
  } else {
    throw std::invalid_argument("invalid sampleBreakMode: " + sampleBreakMode);
  }

  std::cerr << "[LmBlobDataset] process " << worldRank << ": loaded "
            << numTokens << " words, " << sentenceEnds.size()
            << " sentences and " << size() << " batches" << std::endl;
}

int64_t BlobDataset::size() const {
  return batches_.size();
}

std::vector<af::array> BlobDataset::get(const int64_t idx) const {
  const auto& batch = batches_[idx % size()];
  int64_t length = 0;
  for (const auto& pos : batch) {
    length = std::max<int64_t>(length, pos.last - pos.first);
  }
  std::vector<int> buffer(batch.size() * (length + 1), pad_);
  for (int64_t s = 0; s < batch.size(); ++s) {
    const auto& pos = batch[s];
    for (int64_t i = pos.first; i < pos.last + 1; ++i) {
      buffer[s * (length + 1) + (i - pos.first)] = data_[i];
    }
  }
  af::array arr(length + 1, batch.size(), buffer.data());
  std::vector<af::array> sample(2);
  sample[0] = arr(af::seq(0, length - 1), af::span);
  sample[1] = arr(af::seq(1, length), af::span);
  return sample;
}

void BlobDataset::shuffle(uint64_t seed) {
  std::mt19937_64 rng(seed);
  // Deterministic method across compilers.
  for (uint64_t i = size() - 1; i >= 1; --i) {
    std::swap(batches_[i], batches_[rng() % (i + 1)]);
  }
}

} 
}
}