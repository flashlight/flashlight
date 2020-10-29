/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/app/asr/data/Dataset.h"
#include "flashlight/app/asr/data/SpeechSample.h"
#include "flashlight/app/asr/data/Utils.h"

namespace fl {
namespace app {
namespace asr {

class ListFilesDataset : public Dataset {
 public:
  ListFilesDataset(
      const std::string& filenames,
      const lib::text::DictionaryMap& dicts,
      const lib::text::LexiconMap& lexicon,
      int64_t batchSize,
      int worldRank = 0,
      int worldSize = 1,
      bool fallback2Ltr = false,
      bool skipUnk = false,
      const std::string& rootdir = "");

  ~ListFilesDataset() override;

  virtual std::vector<LoaderData> getLoaderData(
      const int64_t idx) const override;

  virtual std::vector<float> loadSound(const std::string& audioHandle) const;

 private:
  std::vector<int64_t> sampleSizeOrder_;
  std::vector<SpeechSample> data_;
  lib::text::LexiconMap lexicon_;
  bool includeWrd_;
  bool fallback2Ltr_;
  bool skipUnk_;

  std::vector<SpeechSampleMetaInfo> loadListFile(const std::string& filename);
};
} // namespace asr
} // namespace app
} // namespace fl
