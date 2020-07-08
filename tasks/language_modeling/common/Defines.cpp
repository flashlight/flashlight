/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common/Defines.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

using namespace fl::lib;

namespace fl {
namespace task {
namespace lm {

// Dictionary
DEFINE_string(dictionary, "", "Path to save the dictionary");
DEFINE_int64(maxwords, 0, "number of workers");
DEFINE_int64(minappearence, 0, "number of workers");

// DATA OPTIONS
DEFINE_string(train, "", "comma-separated list of training data");
DEFINE_string(valid, "", "comma-separated list of valid data");
DEFINE_string(test, "", "comma-separated list of test data");
DEFINE_int64(batchsize, 1, "batch size (per process in distributed training)");

// LEARNING HYPER-PARAMETER OPTIONS
DEFINE_int64(iter, std::numeric_limits<int64_t>::max(), "number of updates");
DEFINE_bool(itersave, false, "save model at each iteration");
DEFINE_double(lr, 1.0, "learning rate");
DEFINE_double(momentum, 0.0, "momentum factor");
DEFINE_double(weightdecay, 0.0, "weight decay (L2 penalty)");
DEFINE_double(lrcrit, 0, "criterion learning rate");
DEFINE_int64(warmup, 1, "the LR warmup parameter, in updates");
DEFINE_int64(
    lr_decay,
    std::numeric_limits<int64_t>::max(),
    "Epoch for the first LR decay");
DEFINE_int64(
    lr_decay_step,
    std::numeric_limits<int64_t>::max(),
    "Epochs for each new LR decay");
DEFINE_double(maxgradnorm, 0, "Clip gradients at value (0 = no clipping)");
DEFINE_double(adambeta1, 0.9, "beta1 in the Adam optimizer");
DEFINE_double(adambeta2, 0.999, "beta2 in the Adam optimizer");
DEFINE_double(optimrho, 0.9, "rho in the optimizer");
DEFINE_double(optimepsilon, 1e-8, "epsilon in the optimizer");
DEFINE_int64(num_labels, 0, "# of classes for target labels");
DEFINE_int64(tokens_per_sample, 1024, "# of tokens per sample");
DEFINE_string(sample_break_mode, "none", "none, eos");
DEFINE_bool(use_dynamic_batching, false, "if or not use dynamic batching");
DEFINE_int64(saveiters, 0, "save every # iterations");

// LR-SCHEDULER OPTIONS
DEFINE_int64(
    stepsize,
    std::numeric_limits<int64_t>::max(),
    "We multiply LR by gamma every stepsize updates");
DEFINE_double(gamma, 1.0, "the LR annealing multiplier");

// OPTIMIZER OPTIONS
DEFINE_string(netoptim, kSGDOptimizer, "optimizer for the network");
DEFINE_string(critoptim, kSGDOptimizer, "optimizer for the criterion");

// RUN OPTIONS
DEFINE_string(datadir, "", "speech data directory");
DEFINE_string(tokensdir, "", "dictionary directory");
DEFINE_string(rundir, "", "experiment root directory");
DEFINE_string(archdir, "", "arch root directory");
DEFINE_string(flagsfile, "", "File specifying gflags");
DEFINE_string(runname, "", "name of current run");
DEFINE_int64(nthread, 1, "specify number of threads for data parallelization");
DEFINE_string(
    tag,
    "",
    "tag this experiment with a particular name (e.g. 'hypothesis1')");
DEFINE_int64(seed, 0, "Manually specify Arrayfire seed.");
DEFINE_int64(
    memstepsize,
    10 * (1 << 20),
    "Minimum allocation size in bytes per array.");
DEFINE_int64(
    reportiters,
    0,
    "number of iterations after which we will run val and save model, \
    if 0 we only do this at end of epoch ");
DEFINE_double(
    pcttraineval,
    100,
    "percentage of training set (by number of utts) to use for evaluation");

// ARCHITECTURE OPTIONS
DEFINE_string(arch, "default", "network architecture");
DEFINE_string(criterion, "adsm", "adsm, ce");
DEFINE_int64(encoderdim, 0, "Dimension of encoded hidden state.");

// ADAPTIVE SOFTMAX OPTIONS
DEFINE_int64(
    adsm_input_size,
    0,
    "input size of AdaptiveSoftMax (i.e. output size of network)");
DEFINE_string(adsm_cutoffs, "", "cutoffs for AdaptiveSoftMax");

// DISTRIBUTED TRAINING
DEFINE_bool(enable_distributed, false, "enable distributed training");
DEFINE_int64(
    world_rank,
    0,
    "rank of the process (Used if rndv_filepath is not empty)");
DEFINE_int64(
    world_size,
    1,
    "total number of the process (Used if rndv_filepath is not empty)");
DEFINE_int64(
    max_devices_per_node,
    8,
    "the maximum number of devices per training node");
DEFINE_string(
    rndv_filepath,
    "",
    "Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");

// FB SPECIFIC
DEFINE_string(target, "tkn", "target feature");
DEFINE_bool(everstoredb, false, "use Everstore db for reading data");
DEFINE_bool(use_memcache, false, "use Memcache for reading data");

} // namespace lm
} // namespace task
} // namespace fl
