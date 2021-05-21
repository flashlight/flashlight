/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file runtime/Helpers.h
 *
 * Reusable helper functions for binary files like Train.cpp. For functions
 * that aren't generic enough to go in `common`, `libraries/common`, etc.
 */

#pragma once

#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "flashlight/fl/flashlight.h"

#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/common/Flags.h"
#include "flashlight/pkg/speech/criterion/criterion.h"
#include "flashlight/pkg/speech/data/ListFileDataset.h"

#include "flashlight/lib/common/String.h"
#include "flashlight/lib/text/dictionary/Utils.h"

namespace fl {
namespace app {
namespace asr {

/**
 * Given a filename, remove any filepath delimiters - returns a contiguous
 * string that won't be subdivided into a filepath.
 */
std::string cleanFilepath(const std::string& inputFileName);

/**
 * Serialize gflags into a buffer.
 *
 * Only serializes gflags that aren't explicitly deprecated.
 */
std::string serializeGflags(const std::string& separator = "\n");

/**
 * Sample indices for the `--pcttraineval` flag.
 */
std::unordered_set<int64_t>
getTrainEvalIds(int64_t dsSize, double pctTrainEval, int64_t seed);

/**
 * Read sample ids from an `af::array`.
 */
std::vector<std::string> readSampleIds(const af::array& arr);

/*
 * Utility function for creating a w2l dataset.
 * From gflags it uses FLAGS_everstoredb and FLAGS_memcache
 * @param inputTransform - a function to featurize input
 * @param targetTransform - a function to featurize target
 * @param wordTransform - a function to featurize words
 * @param padVal - a tuple of padding values when batching input, target, word
 * @param batchingStrategy - batching strategy for the data, for now "none" and
 * "dynamic"
 * @param maxDurationPerBatch - is used for batchingStrategy="dynamic", max
 * total duration in a batch
 */
std::shared_ptr<fl::Dataset> createDataset(
    const std::vector<std::string>& paths,
    const std::string& rootDir = "",
    int batchSize = 1,
    const fl::Dataset::DataTransformFunction& inputTransform = nullptr,
    const fl::Dataset::DataTransformFunction& targetTransform = nullptr,
    const fl::Dataset::DataTransformFunction& wordTransform = nullptr,
    const std::tuple<int, int, int>& padVal =
        std::tuple<int, int, int>{0, -1, -1},
    int worldRank = 0,
    int worldSize = 1,
    const bool allowEmpty = false,
    const std::string& batchingStrategy = kBatchStrategyNone,
    int maxDurationPerBatch = 0);

std::shared_ptr<fl::Dataset> loadPrefetchDataset(
    std::shared_ptr<fl::Dataset> dataset,
    int prefetchThreads,
    bool shuffle,
    int shuffleSeed = 0);

/*
 * Function to parse valid set string describing multiple datasets into a vector
 * Input Format: d1:d1.lst,d2:d2.lst returns {{d1, d1.lst}, {d2, d2.lst}}
 * Input Format: d1.lst,d2.lst returns {{d1.lst, d1.lst}, {d2.lst, d2.lst}}
 */
std::vector<std::pair<std::string, std::string>> parseValidSets(
    const std::string& valid);

} // namespace asr
} // namespace app
} // namespace fl
