/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/text/data/TextDataset.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/lib/text/dictionary/Defines.h"

using fl::lib::text::Dictionary;
using fl::lib::text::PartialFileReader;
using fl::lib::text::Tokenizer;

namespace fl {
namespace pkg {
namespace text {

namespace {

// Maximum number of tokens to keep in memory for each `TextDataset` instance.
// Setting the default value to 10,000,000,000 which requires 40GB in memory,
// since indices are stored as int32.
constexpr size_t kMaxTokenInBuffer = 10000000000;

} // namespace

TextDataset::TextDataset(
    const std::string& dataDirectory,
    const std::string& filenames,
    PartialFileReader& reader,
    const Tokenizer& tokenizer,
    const Dictionary& dictionary,
    int64_t tokensPerSample /* = 1024 */,
    int64_t batchSize /* = 1 */,
    const std::string& sampleBreakMode /* = "none" */,
    bool useDynamicBatching /* = false */)
    : pad_(dictionary.getIndex(fl::lib::text::kPadToken)) {
  /* 1. Read data */
  // data_ will have the following layout:
  // <eos> sentence <eos> sentence <eos> ... <eos> sentence <eos>
  data_.clear();
  data_.reserve(kMaxTokenInBuffer);
  const auto eos = dictionary.getIndex(fl::lib::text::kEosToken);
  data_.push_back(eos);

  // Each pair of indices in sentenceRanges indicates the position in data_ of
  // the 2 <eos> tokens around a given sentence.
  std::vector<std::pair<int64_t, int64_t>> sentenceRanges;
  auto files = lib::split(',', filenames);
  for (const auto& file : files) {
    const auto path = fl::lib::pathsConcat(dataDirectory, file);
    reader.loadFile(path);

    while (reader.hasNextLine()) {
      const auto currentEosPosition = data_.size() - 1;
      if (!sentenceRanges.empty()) {
        sentenceRanges.back().second = currentEosPosition;
      }

      const auto tokens = tokenizer.tokenize(reader.getLine());
      const auto indices = dictionary.mapEntriesToIndices(tokens);
      if (data_.size() + indices.size() > kMaxTokenInBuffer) {
        FL_LOG(INFO) << "[TextDataset] stop loading at 10,000,000,000 tokens";
        break;
      }
      sentenceRanges.emplace_back(currentEosPosition, -1);
      data_.insert(data_.end(), indices.begin(), indices.end());
      data_.push_back(eos);
    }
    if (!sentenceRanges.empty()) {
      sentenceRanges.back().second = data_.size() - 1;
    }
  }
  const int64_t nTokens = data_.size();

  /* 2. Batchify */
  if (batchSize <= 0) {
    throw std::invalid_argument(
        "[TextDataset] BatchSize needs to be positive.");
  }

  if (sampleBreakMode == "none") {
    // Sentences are split into equal size (=`tokensPerSample`)
    // Total tokens per batch is `batchSize` * `tokensPerSample`

    const int64_t nSamples = (nTokens + tokensPerSample - 1) / tokensPerSample;
    const int64_t nBatches = (nSamples + batchSize - 1) / batchSize;
    for (int64_t b = 0; b < nBatches; ++b) {
      const int64_t firstSample = b * batchSize;
      const int64_t lastSample = std::min((b + 1) * batchSize, nSamples);
      std::vector<SamplePosition> batch;
      for (int64_t s = firstSample; s < lastSample; ++s) {
        const int64_t firstToken = s * tokensPerSample;
        const int64_t lastToken = std::min((s + 1) * tokensPerSample, nTokens);
        batch.emplace_back(SamplePosition{firstToken, lastToken - 1});
      }
      batches_.push_back(std::move(batch));
    }
  } else if (sampleBreakMode == "eos") {
    // Each sentence must begin and end in <eos>.
    // Sentences with length > `tokensPerSample` are skipped;
    // Total tokens per batch <= `batchSize` * `tokensPerSample`

    if (useDynamicBatching) {
      // sorting samples by length in ascending order
      std::sort(
          sentenceRanges.begin(),
          sentenceRanges.end(),
          [](const std::pair<int64_t, int64_t>& p1,
             const std::pair<int64_t, int64_t>& p2) {
            return p1.second - p1.first < p2.second - p2.first;
          });
    }

    std::vector<SamplePosition> batch;
    for (int64_t i = 0; i < sentenceRanges.size(); ++i) {
      const auto startPoint = sentenceRanges[i].first;
      const auto endPoint = sentenceRanges[i].second;
      const int64_t sampleSize = endPoint - startPoint + 1;
      batch.emplace_back(SamplePosition{startPoint, endPoint});

      bool isFull;
      if (useDynamicBatching) {
        isFull = sampleSize * (batch.size() + 1) > batchSize * tokensPerSample;
      } else {
        isFull = batch.size() == batchSize;
      }
      if (isFull) {
        batches_.push_back(std::move(batch));
        batch = std::vector<SamplePosition>();
      }
    }
    if (!batch.empty()) {
      batches_.push_back(std::move(batch));
    }
  } else {
    throw std::invalid_argument(
        "Invalid sampleBreakMode: should be none or eos, but it is given " +
        sampleBreakMode);
  }

  FL_LOG(INFO) << "[TextDataset] (" << reader.getRank() << "/"
               << reader.getTotalReaders() << ") Loaded " << nTokens
               << " tokens, " << sentenceRanges.size() << " sentences and "
               << size() << " batches";
}

int64_t TextDataset::size() const {
  return batches_.size();
}

std::vector<af::array> TextDataset::get(const int64_t idx) const {
  const auto& batch = batches_[idx % size()];
  int64_t maxLength = 0;
  for (const auto& pos : batch) {
    maxLength = std::max<int64_t>(maxLength, pos.last - pos.first + 1);
  }
  std::vector<int> buffer(batch.size() * maxLength, pad_);
  for (int64_t i = 0; i < batch.size(); ++i) {
    const auto& pos = batch[i];
    std::memcpy(
        buffer.data() + i * maxLength,
        data_.data() + pos.first,
        sizeof(int) * (pos.last - pos.first + 1));
  }
  return {af::array(maxLength, batch.size(), buffer.data())};
}

void TextDataset::shuffle(uint64_t seed) {
  std::mt19937_64 rng(seed);
  // Deterministic method across compilers.
  for (uint64_t i = size() - 1; i >= 1; --i) {
    std::swap(batches_[i], batches_[rng() % (i + 1)]);
  }
}

} // namespace text
} // namespace pkg
} // namespace fl
