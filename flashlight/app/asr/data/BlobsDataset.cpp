/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>
#include <numeric>
#include <sstream>

#include <glog/logging.h>

#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/data/BlobsDataset.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

using namespace fl::lib;
using fl::lib::text::DictionaryMap;
using fl::lib::text::LexiconMap;

namespace fl {
namespace app {
namespace asr {

BlobsDataset::BlobsDataset(
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
    auto fileSampleInfo = loadBlob(fullpath);
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

BlobsDataset::~BlobsDataset() {
  threadpool_ = nullptr; // join all threads
}

std::vector<LoaderData> BlobsDataset::getLoaderData(const int64_t idx) const {
  std::vector<LoaderData> data(sampleBatches_[idx].size(), LoaderData());
  for (int64_t id = 0; id < sampleBatches_[idx].size(); ++id) {
    auto i = sampleSizeOrder_[sampleBatches_[idx][id]];

    if (!(i >= 0 && i < sampleIndex_.size())) {
      throw std::out_of_range("BlobsDataset::getLoaderData idx out of range");
    }

    data[id].sampleId = std::to_string(sampleIndex_[i]);

    auto rawSample = blobs_.at(blobIndex_[i])->rawGet(sampleIndex_[i]);
    auto& audio_v = rawSample.at(0);
    auto& target_v = rawSample.at(1);

    std::string target((char*)target_v.data(), target_v.size());
    auto transcript = splitOnWhitespace(target, true);

    std::istringstream audiois(
        std::string((char*)audio_v.data(), audio_v.size()));
    data[id].input = loadSound<float>(audiois);
    data[id].targets[kTargetIdx] = wrd2Target(
        transcript, lexicon_, dicts_.at(kTargetIdx), fallback2Ltr_, skipUnk_);

    if (includeWrd_) {
      data[id].targets[kWordIdx] = transcript;
    }
  }
  return data;
}

std::vector<SpeechSampleMetaInfo> BlobsDataset::loadBlob(
    const std::string& filename) {
  std::shared_ptr<fl::BlobDataset> blob =
      std::make_shared<fl::FileBlobDataset>(filename);

  // The format of the list: columns should be space-separated
  // [utterance id] [audio file (full path)] [audio length] [word transcripts]
  std::vector<SpeechSampleMetaInfo> samplesMetaInfo;
  auto curDataSize = blobIndex_.size();
  int64_t idx = curDataSize;
  for (int64_t s = 0; s < blob->size(); s++) {
    auto info = blob->getEntries(s);

    blobIndex_.emplace_back(blobs_.size());
    sampleIndex_.emplace_back(s);

    samplesMetaInfo.emplace_back(SpeechSampleMetaInfo(
        info.at(0).dims.elements(), info.at(1).dims.elements(), idx));

    ++idx;
  }

  blobs_.push_back(blob);

  LOG(INFO) << samplesMetaInfo.size() << " files found. ";

  return samplesMetaInfo;
}
} // namespace asr
} // namespace app
} // namespace fl
