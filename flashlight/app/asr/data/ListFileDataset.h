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

namespace fl {
namespace app {
namespace asr {

/**
 *
 * ListFileDataset class encapsulates the loading of dataset files used in
 * wav2letter++ and makes it easy to work with Dataset classes from flashlight.
 * It accepts a input file consisting of several lines with each row of the
 * form 'utterance_id  input_handle size transcription' where
 * `sample_id` - unique id for the sample
 * `input_handle` - input audio file path.
 * `size` - a real number used for sorting the dataset.
 * `transcription` - word transcrption for this sample
 *
 * It also accepts optional params - `inFeatFunc` and `tgtFeatFunc` are used to
 * specify the featurization for input and target.
 *
 * Example input file format:
 *  train001 /tmp/000000000.flac 100.03  this is sparta
 *  train002 /tmp/000000000.flac 360.57  coca cola
 *  train003 /tmp/000000000.flac 123.53  hello world
 *  train004 /tmp/000000000.flac 999.99  quick brown fox jumped
 *
 *
 * Calling `dataset.get(idx)` returns an af::array vector of size 4 - `input`,
 * `target`, `word_transcription`, `sample_id` in the same order.
 *
 */
class ListFileDataset : public fl::Dataset {
 public:
  explicit ListFileDataset(
      const std::string& filename,
      const DataTransformFunction& inFeatFunc = nullptr,
      const DataTransformFunction& tgtFeatFunc = nullptr,
      const DataTransformFunction& wrdFeatFunc = nullptr);

  int64_t size() const override;

  std::vector<af::array> get(const int64_t idx) const override;

  float getInputSize(const int64_t idx) const;

  int64_t getTargetSize(const int64_t idx) const;

  virtual std::pair<std::vector<float>, af::dim4> loadAudio(
      const std::string& handle) const;

 protected:
  DataTransformFunction inFeatFunc_, tgtFeatFunc_, wrdFeatFunc_;
  int64_t numRows_;
  std::vector<std::string> ids_;
  std::vector<std::string> inputs_;
  std::vector<std::string> targets_;
  std::vector<float> inputSizes_;
  mutable std::vector<int64_t> targetSizesCache_;
};

} // namespace asr
} // namespace app
} // namespace fl
