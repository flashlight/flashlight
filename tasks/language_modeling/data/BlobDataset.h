/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include <vector>

#include <flashlight/flashlight.h>

#include "libraries/language/dictionary/Dictionary.h"
#include "libraries/language/tokenizer/Tokenizer.h"

namespace fl {
namespace task {
namespace lm {

class BlobDataset : public fl::Dataset {
 public:
  explicit BlobDataset(
      const std::string& dataDirectory,
      const std::string& filenames,
      const lib::Dictionary& dictionary,
      int worldRank,
      int worldSize,
      int64_t tokensPerSample = 1024,
      int64_t batchSize = 1,
      int pad = 1,
      int eos = 2,
      const std::string& sampleBreakMode = "none",
      bool useDynamicBatching = false);

  int64_t size() const override;

  std::vector<af::array> get(const int64_t idx) const override;

  void shuffle(uint64_t seed);

 private:
  lib::Dictionary dictionary_;
  lib::Tokenizer tokenizer_;
  int pad_;

  struct SamplePosition {
    int64_t first;
    int64_t last;
  };

  std::vector<int> data_; // eos prepended, so all indices shifted by 1
  std::vector<std::vector<SamplePosition>> batches_;
};

} 
}
}