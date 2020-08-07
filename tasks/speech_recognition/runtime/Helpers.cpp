/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/tasks/speech_recognition/runtime/Helpers.h"

#include <glog/logging.h>
#include <random>

#include "flashlight/extensions/common/DistributedUtils.h"
#include "flashlight/libraries/common/System.h"

using namespace fl::ext;
using namespace fl::lib;

namespace fl {
namespace tasks {
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
    LOG(FATAL) << "EverstoreDataset not supported: "
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
} // namespace asr
} // namespace tasks
} // namespace fl
