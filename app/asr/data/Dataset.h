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

#include "flashlight/flashlight/flashlight.h"

#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/app/asr/data/Featurize.h"

namespace fl {
namespace app {
namespace asr {

class Dataset : public fl::Dataset {
 public:
  Dataset(
      const lib::text::DictionaryMap& dicts,
      int64_t batchsize,
      int worldrank = 0,
      int worldsize = 1);

  int64_t size() const override;

  // NB: get() is thread-hostile if FLAGS_nthread > 0: it must be called
  // from only one thread. In this case, it also expects calls to be sequential,
  // or else samples will be loaded into cache unnecessarily and discarded.
  std::vector<af::array> get(const int64_t idx) const override;

  virtual std::vector<LoaderData> getLoaderData(const int64_t idx) const = 0;

  int64_t getGlobalBatchIdx(const int64_t idx);

  FeatureData getFeatureData(const int64_t idx) const;

  FeatureData getFeatureDataAndPrefetch(const int64_t idx) const;

  void shuffle(int seed);

 protected:
  lib::text::DictionaryMap dicts_;

  int64_t sampleCount_; // Num individual samples in the dataset before batching
  int64_t batchSize_;

  // Note, if worldSize = N, then worldRank should be in [0, N)
  int64_t worldRank_; // The GPU id for which this Dataset is being used
  int64_t worldSize_; // Total number of parallel GPUs/ CPUs used in training

  // used if FLAGS_nthread > 1
  std::unique_ptr<fl::ThreadPool> threadpool_;
  mutable std::unordered_map<int64_t, std::future<FeatureData>> prefetchCache_;

  std::vector<std::vector<int64_t>> sampleBatches_;
};

// Abstract class which defines an interface to pack samples
// into batches
class BatchPacker {
 public:
  virtual ~BatchPacker() {}
  virtual std::vector<std::vector<int64_t>> getBatches(
      int64_t numSamples,
      int64_t seed) const = 0;
};

// Implementation which packs the samples into batches in Round Robin
// order.
class RoundRobinBatchPacker : public BatchPacker {
 public:
  RoundRobinBatchPacker(int64_t batchSize, int64_t worldSize, int64_t worldRank)
      : batchSize_(batchSize), worldSize_(worldSize), worldRank_(worldRank) {}

  // Use seed < 0, for no shuffling of the samples
  virtual std::vector<std::vector<int64_t>> getBatches(
      int64_t numSamples,
      int64_t seed) const override;

 private:
  int64_t batchSize_;
  int64_t worldSize_;
  int64_t worldRank_;
};
} // namespace asr
} // namespace app
} // namespace fl
