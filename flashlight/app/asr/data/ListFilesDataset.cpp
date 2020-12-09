/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>
#include <numeric>

#include <glog/logging.h>

#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/data/ListFilesDataset.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

using namespace fl::lib;
using fl::lib::text::DictionaryMap;
using fl::lib::text::LexiconMap;

namespace fl {
namespace app {
namespace asr {

ListFilesDataset::ListFilesDataset(
    const std::string& filenames,
    const DictionaryMap& dicts,
    const LexiconMap& lexicon,
    int64_t batchSize /* = 1 */,
    int worldRank /* = 0 */,
    int worldSize /* = 1 */,
    bool fallback2Ltr /* = false */,
    bool skipUnk /* = false */,
    const std::string& rootdir /* = "" */)
    : Dataset(dicts, batchSize, worldRank, worldSize),
      lexicon_(lexicon),
      fallback2Ltr_(fallback2Ltr),
      skipUnk_(skipUnk) {
  includeWrd_ = (dicts.find(kWordIdx) != dicts.end());

  LOG_IF(FATAL, dicts.find(kTargetIdx) == dicts.end())
      << "Target dictionary does not exist";

  auto filesVec = lib::split(',', filenames);
  std::vector<SpeechSampleMetaInfo> speechSamplesMetaInfo;
  for (const auto& f : filesVec) {
    auto fullpath = pathsConcat(rootdir, trim(f));
    auto fileSampleInfo = loadListFile(fullpath);
    speechSamplesMetaInfo.insert(
        speechSamplesMetaInfo.end(),
        fileSampleInfo.begin(),
        fileSampleInfo.end());
  }

  filterSamples(
      speechSamplesMetaInfo,
      FLAGS_minisz,
      FLAGS_maxisz,
      FLAGS_mintsz,
      FLAGS_maxtsz);
  sampleCount_ = speechSamplesMetaInfo.size();
  sampleSizeOrder_ = sortSamples(
      speechSamplesMetaInfo,
      FLAGS_dataorder,
      FLAGS_inputbinsize,
      FLAGS_outputbinsize);

  shuffle(-1);
  LOG(INFO) << "Total batches (i.e. iters): " << sampleBatches_.size();
}

ListFilesDataset::~ListFilesDataset() {
  threadpool_ = nullptr; // join all threads
}

std::vector<LoaderData> ListFilesDataset::getLoaderData(
    const int64_t idx) const {
  std::vector<LoaderData> data(sampleBatches_[idx].size(), LoaderData());
  for (int64_t id = 0; id < sampleBatches_[idx].size(); ++id) {
    auto i = sampleSizeOrder_[sampleBatches_[idx][id]];

    if (!(i >= 0 && i < data_.size())) {
      throw std::out_of_range(
          "ListFilesDataset::getLoaderData idx out of range");
    }

    data[id].sampleId = data_[i].getSampleId();
    data[id].input = loadSound(data_[i].getAudioFile());
    data[id].targets[kTargetIdx] = wrd2Target(
        data_[i].getTranscript(),
        lexicon_,
        dicts_.at(kTargetIdx),
        fallback2Ltr_,
        skipUnk_);

    if (includeWrd_) {
      data[id].targets[kWordIdx] = data_[i].getTranscript();
    }
  }
  return data;
}

std::vector<float> ListFilesDataset::loadSound(
    const std::string& audioHandle) const {
  return fl::app::asr::loadSound<float>(audioHandle);
}

std::vector<SpeechSampleMetaInfo> ListFilesDataset::loadListFile(
    const std::string& filename) {
  std::ifstream infile(filename);

  LOG_IF(FATAL, !infile) << "Could not read file '" << filename << "'";

  // The format of the list: columns should be space-separated
  // [utterance id] [audio file (full path)] [audio length] [word transcripts]
  std::string line;
  std::vector<SpeechSampleMetaInfo> samplesMetaInfo;
  auto curDataSize = data_.size();
  int64_t idx = curDataSize;
  while (std::getline(infile, line)) {
    auto tokens = splitOnWhitespace(line, true);

    LOG_IF(FATAL, tokens.size() < 3) << "Cannot parse " << line;

    data_.emplace_back(SpeechSample(
        tokens[0],
        tokens[1],
        std::vector<std::string>(tokens.begin() + 3, tokens.end())));

    auto audioLength = std::stod(tokens[2]);
    auto targets = wrd2Target(
        data_.back().getTranscript(),
        lexicon_,
        dicts_.at(kTargetIdx),
        fallback2Ltr_,
        skipUnk_);

    samplesMetaInfo.emplace_back(
        SpeechSampleMetaInfo(audioLength, targets.size(), idx));

    ++idx;
  }

  if (samplesMetaInfo.size() < 1) {
    throw std::runtime_error("Train files not found from " + filename);
  }

  LOG(INFO) << samplesMetaInfo.size() << " files found. ";

  return samplesMetaInfo;
}
} // namespace asr
} // namespace app
} // namespace fl
