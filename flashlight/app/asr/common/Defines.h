/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gflags/gflags.h>
#include <memory>
#include <unordered_map>

#define FL_TASK_ASR_VERSION "0.1"

namespace fl {
namespace app {
namespace asr {

namespace detail {

using DeprecatedFlagsMap = std::unordered_map<std::string, std::string>;

/**
 * Creates and maintains a map of deprecated flags. The map takes
 * a deprecated flag name to a new flag name; for instance, the entry:
 * ---> {myOldFlag, myNewFlag}
 * corresponds to the deprecation of myOldFlag
 */
DeprecatedFlagsMap& getDeprecatedFlags();

// Adds a flag to the global deprecated map
void addDeprecatedFlag(
    const std::string& depreactedFlagName,
    const std::string& newFlagName);

// Whether the flag has been explicitly set either from the cmd line or
// de-serialization
bool isFlagSet(const std::string& name);

} // namespace detail

/**
 * Globally-accessible and recommended to be called immediately after gflags
 * have been parsed and initialized. Does a few things:
 * - Sets the value of the new flag to be the value of the old flag
 * - Displays a message indicating that the old flag is deprecated and the new
 * flag shoule be used.
 *
 * Behavior is as follows:
 * - Throws if the user set both the deprecated flag and the new flag.
 * - Sets the new flag equal to the deprecated flag if the user only set the
 * deprecated flag.
 * - Does nothing if the user set neither the new nor the deprecated flag, or if
 * the user correctly set only the new flag and not the deprecated flag.
 */
void handleDeprecatedFlags();

/**
 * Deprecate a command line flag.
 *
 * USAGE:
 *   DEPRECATE_FLAGS(myOldFlagName, my_new_flag_name)
 */
#define DEPRECATE_FLAGS(DEPRECATED, NEW) \
  detail::addDeprecatedFlag(#DEPRECATED, #NEW);

// Dataset indices
// If a new field is added, `kNumDataIdx` should be modified accordingly.
constexpr size_t kInputIdx = 0;
constexpr size_t kTargetIdx = 1;
constexpr size_t kWordIdx = 2;
constexpr size_t kSampleIdx = 3;
constexpr size_t kNumDataIdx = 4; // total number of dataset indices

// Various constants used in asr task
constexpr const char* kTrainMode = "train";
constexpr const char* kContinueMode = "continue";
constexpr const char* kForkMode = "fork";
constexpr const char* kGflags = "gflags";
constexpr const char* kCommandLine = "commandline";
constexpr const char* kUserName = "username";
constexpr const char* kHostName = "hostname";
constexpr const char* kTimestamp = "timestamp";
constexpr const char* kRunIdx = "runIdx";
constexpr const char* kRunPath = "runPath";
constexpr const char* kProgramName = "programname";
constexpr const char* kEpoch = "epoch";
constexpr const char* kUpdates = "updates";
constexpr const char* kSGDOptimizer = "sgd";
constexpr const char* kAdamOptimizer = "adam";
constexpr const char* kRMSPropOptimizer = "rmsprop";
constexpr const char* kAdadeltaOptimizer = "adadelta";
constexpr const char* kAdagradOptimizer = "adagrad";
constexpr const char* kAMSgradOptimizer = "amsgrad";
constexpr const char* kNovogradOptimizer = "novograd";
constexpr const char* kCtcCriterion = "ctc";
constexpr const char* kAsgCriterion = "asg";
constexpr const char* kSeq2SeqCriterion = "seq2seq";
constexpr const char* kTransformerCriterion = "transformer";
constexpr int kTargetPadValue = -1;

// Feature params
constexpr int kLifterParam = 22;
constexpr int kPrefetchSize = 2;

constexpr const char* kEosToken = "$";
constexpr const char* kBlankToken = "#";
constexpr const char* kSilToken = "|";
} // namespace asr
} // namespace app
} // namespace fl
