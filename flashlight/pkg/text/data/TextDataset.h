/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include <vector>

#include "flashlight/fl/flashlight.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/tokenizer/PartialFileReader.h"
#include "flashlight/lib/text/tokenizer/Tokenizer.h"

namespace fl {
namespace pkg {
namespace text {

/**
 * TextDataset prepares text data for LM training. It returns a single tensor of
 * the indices for a given batched text. The indices are the token ID of each
 * token given a certain Dictionary. Each sentence is padded with <eos> on both
 * ends.
 *
 * @param dataDirectory A prefix for the files to read
 * @param filenames A comma separated list of files with training data
 *
 * @param partialFileReader A reader used to part of a text file line by line
 * @param tokenizer A tokenizer to tokenize lines of sentences to tokens
 * @param dictionary A dictionary to map tokens to their indices
 *
 * @param tokensPerSample Maximum number of allowed tokens of one sample
 * @param batchSize The number of samples in a batch
 *
 * @param sampleBreakMode Strategy to break sentences and form batches:
 * - "none": Break sentences into equal length, regardless of <eos>
 *           Total tokens per batch is `batchSize` * `tokensPerSample`
 * - "eos": Each sentence is a sample padded with <eos> on both ends.
 *          Sentences with length > `tokensPerSample` are skipped;
 *          Total tokens per batch <= `batchSize` * `tokensPerSample`
 * @param useDynamicBatching Use dynamic batching when `sampleBreakMode`="eos".
 * In this case, `batchsize` is ignored and as many sentences as possible are
 * included in each batch. All samples are padded with token <pad> to the length
 * of the longest one in a certain batch. To better fit more samples in each
 * batch, samples are sorted by length.
 */

class TextDataset : public fl::Dataset {
 public:
  TextDataset(
      const std::string& dataDirectory,
      const std::string& filenames,
      fl::lib::text::PartialFileReader& reader,
      const fl::lib::text::Tokenizer& tokenizer,
      const fl::lib::text::Dictionary& dictionary,
      int64_t tokensPerSample = 1024,
      int64_t batchSize = 1,
      const std::string& sampleBreakMode = "none",
      bool useDynamicBatching = false);

  int64_t size() const override;

  std::vector<af::array> get(const int64_t idx) const override;

  void shuffle(uint64_t seed);

 private:
  int pad_;

  struct SamplePosition {
    int64_t first;
    int64_t last;
  };

  std::vector<int> data_; // eos prepended, so all indices shifted by 1
  std::vector<std::vector<SamplePosition>> batches_;
};

} // namespace text
} // namespace pkg
} // namespace fl
