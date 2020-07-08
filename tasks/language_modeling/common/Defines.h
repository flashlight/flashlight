/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "libraries/language/dictionary/Defines.h"

#include <memory>
#include <unordered_map>

#include <gflags/gflags.h>

#define FL_TASK_LM_VERSION "0.1"

namespace fl {
namespace task {
namespace lm {

constexpr int kPadIdx = 1;
constexpr int kEosIdx = 2;
constexpr int kUnkIdx = 3;

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

/* ========== DATA OPTIONS ========== */
DECLARE_string(dictionary);
DECLARE_int64(maxwords);
DECLARE_int64(minappearence);


/* ========== DATA OPTIONS ========== */
DECLARE_string(train);
DECLARE_string(valid);
DECLARE_string(test);
DECLARE_int64(batchsize);
DECLARE_int64(num_labels);
DECLARE_int64(tokens_per_sample);
DECLARE_string(sample_break_mode);
DECLARE_bool(use_dynamic_batching);


/* ========== LEARNING HYPER-PARAMETER OPTIONS ========== */
DECLARE_int64(iter);
DECLARE_bool(itersave);
DECLARE_double(lr);
DECLARE_double(momentum);
DECLARE_double(weightdecay);
DECLARE_double(lrcrit);
DECLARE_int64(warmup);
DECLARE_int64(lr_decay);
DECLARE_int64(lr_decay_step);
DECLARE_double(maxgradnorm);
DECLARE_double(adambeta1); // TODO rename into optim beta1
DECLARE_double(adambeta2); // TODO rename into optim beta2
DECLARE_double(optimrho);
DECLARE_double(optimepsilon);
DECLARE_int64(saveiters);


/* ========== LR-SCHEDULER OPTIONS ========== */
DECLARE_int64(stepsize);
DECLARE_double(gamma);

/* ========== OPTIMIZER OPTIONS ========== */
DECLARE_string(netoptim);
DECLARE_string(critoptim);

/* ========== RUN OPTIONS ========== */
DECLARE_string(datadir);
DECLARE_string(tokensdir);
DECLARE_string(rundir);
DECLARE_string(archdir);
DECLARE_string(flagsfile);
DECLARE_string(runname);
DECLARE_int64(nthread);
DECLARE_string(tag);
DECLARE_int64(seed);
DECLARE_int64(memstepsize);
DECLARE_int64(reportiters);
DECLARE_double(pcttraineval);

/* ========== ARCHITECTURE OPTIONS ========== */
DECLARE_string(arch);
DECLARE_string(criterion);
DECLARE_int64(encoderdim);

/* ========== ADAPTIVE SOFTMAX OPTIONS ========== */
DECLARE_int64(adsm_input_size);
DECLARE_string(adsm_cutoffs);

/* ========== DISTRIBUTED TRAINING ========== */
DECLARE_bool(enable_distributed);
DECLARE_int64(world_rank);
DECLARE_int64(world_size);
DECLARE_int64(max_devices_per_node);
DECLARE_string(rndv_filepath);

/* ========== FB SPECIFIC ========== */
DECLARE_string(target);
DECLARE_bool(everstoredb);
DECLARE_bool(use_memcache);

} // namespace lm
} // namespace task
} // namespace fl
