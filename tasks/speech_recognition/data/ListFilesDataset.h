/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/tasks/speech_recognition/data/Dataset.h"
#include "flashlight/tasks/speech_recognition/data/SpeechSample.h"
#include "flashlight/tasks/speech_recognition/data/Utils.h"

namespace fl {
namespace tasks {
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
} // namespace tasks
} // namespace fl
