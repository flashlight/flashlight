/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/flags/TrainFlags.h"
#include <limits>

namespace fl {
namespace app {
namespace asr {

// DATA OPTIONS
DEFINE_string(train, "", "comma-separated list of training data");
DEFINE_string(valid, "", "comma-separated list of valid data");
DEFINE_int64(batchsize, 1, "batch size (per process in distributed training)");
DEFINE_int64(
    validbatchsize,
    -1,
    "batch size (per process in distributed training) for the valid data, if -1 then use train batchsize");
DEFINE_bool(noresample, false, "do not resample training data");
DEFINE_double(
    sampletarget,
    0.0,
    "probability [0.0, 1.0] for randomly sampling targets from a lexicon if there are multiple mappings from a word");

// TRAINING
DEFINE_bool(lrcosine, false, "use cosine learning rate schedule");
DEFINE_int64(iter, std::numeric_limits<int64_t>::max(), "number of updates");
DEFINE_bool(itersave, false, "save model at each iteration");
DEFINE_double(lr, 1.0, "learning rate");
DEFINE_double(momentum, 0.0, "momentum factor");
DEFINE_double(weightdecay, 0.0, "weight decay (L2 penalty)");
DEFINE_double(lrcrit, 0, "criterion learning rate");
DEFINE_int64(warmup, 1, "the LR warmup parameter, in updates");
DEFINE_int64(
    saug_start_update,
    -1,
    "Use SpecAugment starting at the update number inputted. -1 means no SpecAugment");
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

// LR-SCHEDULER OPTIONS
DEFINE_int64(
    stepsize,
    std::numeric_limits<int64_t>::max(),
    "We multiply LR by gamma every stepsize updates");
DEFINE_double(gamma, 1.0, "the LR annealing multiplier");

// OPTIMIZER OPTIONS
DEFINE_string(netoptim, kSGDOptimizer, "optimizer for the network");
DEFINE_string(critoptim, kSGDOptimizer, "optimizer for the criterion");

// SPECAUGMENT OPTIONS
DEFINE_int64(saug_fmaskf, 27, "Max number of frequency bands that are masked");
DEFINE_int64(saug_fmaskn, 2, "Number of frequency masks");
DEFINE_int64(saug_tmaskt, 100, "Max number of timesteps that are masked");
DEFINE_double(
    saug_tmaskp,
    1.0,
    "Max proportion of the input sequence (1.0 is 100%) that can be masked in time");
DEFINE_int64(saug_tmaskn, 2, "Number of time masks");

// RUN OPTIONS
DEFINE_string(rundir, "", "experiment root directory");
DEFINE_string(archdir, "", "arch root directory");
DEFINE_string(runname, "", "name of current run");
DEFINE_int64(nthread, 1, "specify number of threads for data parallelization");
DEFINE_string(
    tag,
    "",
    "tag this experiment with a particular name (e.g. 'hypothesis1')");
DEFINE_int64(
    reportiters,
    0,
    "number of iterations after which we will run val and save model, \
    if 0 we only do this at end of epoch ");
DEFINE_double(
    pcttraineval,
    100,
    "percentage of training set (by number of utts) to use for evaluation");

DEFINE_bool(
    fl_benchmark_mode,
    true,
    "Sets flashlight benchmark mode, which dynamically "
    "benchmarks various operations based on their empirical performance on "
    "current hardware throughout training");
DEFINE_string(
    fl_optim_mode,
    "",
    "Sets the flashlight optimization mode. "
    "Optim modes can be O1, O2, or O3.");

// MIXED PRECISION OPTIONS
DEFINE_bool(
    fl_amp_use_mixed_precision,
    false,
    "Use mixed precision for training - scale loss and gradients up and down "
    "by a scale factor that changes over time. If no fl optim mode is "
    "specified with --fl_optim_mode when passing this flag, automatically "
    "sets the optim mode to O1.");
DEFINE_uint64(
    fl_amp_scale_factor,
    4096,
    "Starting scale factor to use for loss scaling "
    " with mixed precision training");
DEFINE_uint64(
    fl_amp_scale_factor_update_interval,
    2000,
    "Update interval for adjusting loss scaling in mixed precision training");
DEFINE_uint64(
    fl_amp_max_scale_factor,
    32000,
    "Maximum value for the loss scale factor in mixed precision training");

// ARCHITECTURE OPTIONS
DEFINE_string(arch, "default", "network architecture");
DEFINE_int64(encoderdim, 0, "Dimension of encoded hidden state.");

// Seq2Seq Transformer decoder
DEFINE_int64(
    am_decoder_tr_layers,
    1,
    "s2s transformer decoder: number of layers");
DEFINE_double(am_decoder_tr_dropout, 0.0, "s2s transformer decoder: dropout");
DEFINE_double(
    am_decoder_tr_layerdrop,
    0.0,
    "s2s transformer decoder: layerdrop");

DEFINE_double(
    smoothingtemperature,
    1.0,
    "smoothening the probability distribution in seq2seq decoder");
DEFINE_int32(
    attentionthreshold,
    std::numeric_limits<int>::max(),
    "hard attention limit");

// ASG OPTIONS
DEFINE_int64(linseg, 0, "# of updates of LinSeg to init transitions for ASG");
DEFINE_double(linlr, -1.0, "LinSeg learning rate (if < 0, use lr)");
DEFINE_double(
    linlrcrit,
    -1.0,
    "LinSeg criterion learning rate (if < 0, use lrcrit)");
DEFINE_double(
    transdiag,
    0.0,
    "Initial value along diagonal of ASG transition matrix");

DEFINE_int64(
    pctteacherforcing,
    100,
    "Percentage of steps to train using teacher forcing");
DEFINE_string(
    samplingstrategy,
    "rand",
    "Sampling strategy to use when pctteacherforcing < 100. rand or model");
DEFINE_double(
    labelsmooth,
    0.0,
    "Fraction to smooth targets with uniform distribution.");
DEFINE_bool(inputfeeding, false, "feed encoder summary to the decoder RNN");
DEFINE_string(attention, "content", "attention type");
DEFINE_string(attnWindow, "no", "attention window type");
DEFINE_int64(attndim, 0, "Dimension of neural location attention");
DEFINE_int64(
    attnconvchannel,
    0,
    "Number of convolutional channels for location attention");
DEFINE_int64(attnconvkernel, 0, "Kernel width for location attention");
DEFINE_int64(numattnhead, 8, "number of heads for multihead attention");
DEFINE_int64(leftWindowSize, 50, "left median window width");
DEFINE_int64(rightWindowSize, 50, "right median window width");
DEFINE_int64(
    maxsil,
    50,
    "maximum number of leading silence frames for the step window");
DEFINE_int64(
    minsil,
    0,
    "minimum number of leading silence frames for the step window");
DEFINE_double(
    maxrate,
    10,
    "maximum ratio between the transcript and the encoded input lengths for the step window");
DEFINE_double(
    minrate,
    3,
    "minimum ratio between the transcript and the encoded input lengths for the step window");
DEFINE_int64(
    softwoffset,
    10,
    "offset for the soft window center (= offset + step * rate)");
DEFINE_double(
    softwrate,
    5,
    "moving rate for the soft window center (= offset + step * rate)");
DEFINE_double(
    softwstd,
    5,
    "std for the soft window shape (=exp(-(t - center)^2 / (2 * std^2)))");
DEFINE_bool(trainWithWindow, false, "use window in training");
DEFINE_int64(
    pretrainWindow,
    0,
    "use window in training for pretrainWindow in updates");
DEFINE_double(gumbeltemperature, 1.0, "temperature in gumbel softmax");
DEFINE_int64(decoderrnnlayer, 1, "The number of decoder rnn layers.");
DEFINE_int64(decoderattnround, 1, "The number of decoder attention rounds.");
DEFINE_double(decoderdropout, 0.0, "decoder dropout");
} // namespace asr
} // namespace app
} // namespace fl
