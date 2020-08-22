/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file runtime/Utils.h
 *
 * Reusable helper functions for binary files like Train.cpp. For functions
 * that aren't generic enough to go in `common`, `libraries/common`, etc.
 */

#pragma once

#include <unordered_set>

#include <flashlight/flashlight.h>

#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/data/BlobsDataset.h"
#include "flashlight/app/asr/data/ListFileDataset.h"
#include "flashlight/app/asr/data/ListFilesDataset.h"

#include "flashlight/lib/common/String.h"
#include "flashlight/lib/text/dictionary/Utils.h"

namespace fl {
namespace app {
namespace asr {

/**
 * Create the path to save checkpoints and logs of an experiments.
 */
std::string newRunPath(
    const std::string& root,
    const std::string& runname = "",
    const std::string& tag = "");

/**
 * Get a certain checkpoint by `runidx`.
 */
std::string
getRunFile(const std::string& name, int runidx, const std::string& runpath);

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

/**
 * Create dataset.
 */
std::shared_ptr<Dataset> createDataset(
    const std::string& path,
    const lib::text::DictionaryMap& dicts,
    const lib::text::LexiconMap& lexicon = lib::text::LexiconMap(),
    int batchSize = 1,
    int worldRank = 0,
    int worldSize = 1,
    bool fallback2Ltr = true,
    bool skipUnk = true);
} // namespace asr
} // namespace app
} // namespace fl
