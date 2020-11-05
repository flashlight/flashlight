/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/runtime/Helpers.h"

#include <random>

#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/lib/common/System.h"

using namespace fl::ext;
using namespace fl::lib;
using fl::lib::text::DictionaryMap;
using fl::lib::text::LexiconMap;

namespace fl {
namespace app {
namespace asr {

template <class T>
std::vector<std::string> afMatrixToStrings(const af::array& arr, T terminator) {
  int L = arr.dims(0); // padded length of string
  int N = arr.dims(1); // number of strings
  std::vector<std::string> result;
  auto values = afToVector<T>(arr);
  for (int i = 0; i < N; ++i) {
    const T* row = &values[i * L];
    int len = 0;
    while (len < L && row[len] != terminator) {
      ++len;
    }
    result.emplace_back(row, row + len);
  }
  return result;
}

std::string newRunPath(
    const std::string& root,
    const std::string& runname /* = "" */,
    const std::string& tag /* = "" */) {
  std::string dir = "";
  if (runname.empty()) {
    auto dt = getCurrentDate();
    std::string tm = getCurrentTime();
    replaceAll(tm, ":", "-");
    dir += (dt + "_" + tm + "_" + getEnvVar("HOSTNAME", "unknown_host") + "_");

    // Unique hash based on config
    auto hash = std::hash<std::string>{}(serializeGflags());
    dir += std::to_string(hash);

  } else {
    dir += runname;
  }
  if (!tag.empty()) {
    dir += "_" + tag;
  }
  return pathsConcat(root, dir);
}

std::string
getRunFile(const std::string& name, int runidx, const std::string& runpath) {
  auto fname = format("%03d_%s", runidx, name.c_str());
  return pathsConcat(runpath, fname);
};

std::string cleanFilepath(const std::string& in) {
  std::string replace = in;
  std::string sep = "/";
#ifdef _WIN32
  sep = "\\";
#endif
  replaceAll(replace, sep, "#");
  return replace;
}

std::string serializeGflags(const std::string& separator /* = "\n" */) {
  std::stringstream serialized;
  std::vector<gflags::CommandLineFlagInfo> allFlags;
  gflags::GetAllFlags(&allFlags);
  std::string currVal;
  auto& deprecatedFlags = detail::getDeprecatedFlags();
  for (auto itr = allFlags.begin(); itr != allFlags.end(); ++itr) {
    // Check if the flag is deprecated - if so, skip it
    if (deprecatedFlags.find(itr->name) == deprecatedFlags.end()) {
      gflags::GetCommandLineOption(itr->name.c_str(), &currVal);
      serialized << "--" << itr->name << "=" << currVal << separator;
    }
  }
  return serialized.str();
}

std::unordered_set<int64_t>
getTrainEvalIds(int64_t dsSize, double pctTrainEval, int64_t seed) {
  std::mt19937_64 rng(seed);
  std::bernoulli_distribution dist(pctTrainEval / 100.0);
  std::unordered_set<int64_t> result;
  for (int64_t i = 0; i < dsSize; ++i) {
    if (dist(rng)) {
      result.insert(i);
    }
  }
  return result;
}

std::vector<std::string> readSampleIds(const af::array& arr) {
  return afMatrixToStrings<int>(arr, -1);
}

std::shared_ptr<Dataset> createDataset(
    const std::string& path,
    const DictionaryMap& dicts,
    const LexiconMap& lexicon /* = LexiconMap() */,
    int batchSize /* = 1 */,
    int worldRank /* = 0 */,
    int worldSize /* = 1 */,
    bool fallback2Ltr /* = true */,
    bool skipUnk /* = true */) {
  std::shared_ptr<Dataset> ds;
  if (FLAGS_everstoredb) {
#ifdef FL_BUILD_FB_DEPENDENCIES
    EverstoreDataset::init(); // Required for everstore client
    ds = std::make_shared<EverstoreDataset>(
        path,
        dicts,
        lexicon,
        batchSize,
        worldRank,
        worldSize,
        fallback2Ltr,
        skipUnk,
        FLAGS_datadir,
        FLAGS_use_memcache);
#else
    FL_LOG(fl::FATAL) << "EverstoreDataset not supported: "
                      << "build with -DFL_BUILD_FB_DEPENDENCIES";
#endif
  } else if (FLAGS_blobdata) {
    ds = std::make_shared<BlobsDataset>(
        path,
        dicts,
        lexicon,
        batchSize,
        worldRank,
        worldSize,
        fallback2Ltr,
        skipUnk,
        FLAGS_datadir);
  } else {
    ds = std::make_shared<ListFilesDataset>(
        path,
        dicts,
        lexicon,
        batchSize,
        worldRank,
        worldSize,
        fallback2Ltr,
        skipUnk,
        FLAGS_datadir);
  }

  return ds;
}

std::shared_ptr<fl::Dataset> createDataset(
    const std::vector<std::string>& paths,
    const std::string& rootDir /* = "" */,
    int batchSize /* = 1 */,
    const fl::Dataset::DataTransformFunction& inputTransform /* = nullptr */,
    const fl::Dataset::DataTransformFunction& targetTransform /* = nullptr */,
    const fl::Dataset::DataTransformFunction& wordTransform /* = nullptr */,
    const std::tuple<int, int, int>& padVal /* = {0, -1, -1} */,
    int worldRank /* = 0 */,
    int worldSize /* = 1 */) {
  std::vector<std::shared_ptr<const fl::Dataset>> allListDs;
  std::vector<float> sizes;
  for (auto& path : paths) {
    auto curListDs = std::make_shared<ListFileDataset>(
        pathsConcat(rootDir, path),
        inputTransform,
        targetTransform,
        wordTransform);

    allListDs.emplace_back(curListDs);
    sizes.reserve(sizes.size() + curListDs->size());
    for (int64_t i = 0; i < curListDs->size(); ++i) {
      sizes.push_back(curListDs->getInputSize(i));
    }
  }

  // Order Dataset
  std::vector<int64_t> sortedIds(sizes.size());
  std::iota(sortedIds.begin(), sortedIds.end(), 0);
  auto cmp = [&sizes](const int64_t& l, const int64_t& r) {
    return sizes[l] > sizes[r];
  };
  std::stable_sort(sortedIds.begin(), sortedIds.end(), cmp);

  auto concatListDs = std::make_shared<fl::ConcatDataset>(allListDs);

  auto sortedDs =
      std::make_shared<fl::ResampleDataset>(concatListDs, sortedIds);

  // Partition the dataset and distribute
  auto partitions = fl::partitionByRoundRobin(
      sortedDs->size(), worldRank, worldSize, batchSize);
  auto paritionDs = std::make_shared<fl::ResampleDataset>(sortedDs, partitions);

  // Batch the dataset
  int inPad, tgtPad, wrdPad;
  std::tie(inPad, tgtPad, wrdPad) = padVal;
  return std::make_shared<fl::BatchDataset>(
      paritionDs,
      batchSize,
      fl::BatchDatasetPolicy::INCLUDE_LAST,
      std::vector<fl::Dataset::BatchFunction>{
          [inPad](const std::vector<af::array>& arr) {
            return fl::join(arr, inPad, 3);
          },
          [tgtPad](const std::vector<af::array>& arr) {
            return fl::join(arr, tgtPad, 1);
          },
          [wrdPad](const std::vector<af::array>& arr) {
            return fl::join(arr, wrdPad, 1);
          },
          [](const std::vector<af::array>& arr) {
            return fl::join(arr, 0, 1);
          }});
}

std::shared_ptr<fl::Dataset> loadPrefetchDataset(
    std::shared_ptr<fl::Dataset> dataset,
    int prefetchThreads,
    bool shuffle,
    int shuffleSeed /*= 0 */) {
  if (shuffle) {
    dataset = std::make_shared<fl::ShuffleDataset>(dataset, shuffleSeed);
  }
  if (prefetchThreads > 0) {
    dataset = std::make_shared<fl::PrefetchDataset>(
        dataset, prefetchThreads, prefetchThreads /* prefetch size */);
  }
  return dataset;
}

} // namespace asr
} // namespace app
} // namespace fl
