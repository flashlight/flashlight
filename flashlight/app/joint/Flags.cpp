/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/joint/Trainer.h"

using namespace fl::ext;
using namespace fl::lib;

namespace fl {
namespace app {
namespace joint {

/* ================================ FLAGS ================================ */

/* CRITERION OPTIONS */
DEFINE_string(
    loss_type,
    "adsm",
    "Loss type during optimization. \
    Supported for now: adaptive softmax (adsm) and cross entropy (ce).");
DEFINE_int64(
    loss_adsm_input_size,
    0,
    "Input size of AdaptiveSoftMax (i.e. output size of network).");
DEFINE_string(
    loss_adsm_cutoffs,
    "",
    "Cutoffs for AdaptiveSoftMax comma separated.");

/* DISTRIBUTED TRAINING */
DEFINE_bool(distributed_enable, false, "Enable distributed training.");
DEFINE_int64(
    distributed_world_rank,
    0,
    "Distributed training. Rank of the process (Used if rndv_filepath is not empty).");
DEFINE_int64(
    distributed_world_size,
    1,
    "Distributed training. Total number of the process (Used if rndv_filepath is not empty).");
DEFINE_int64(
    distributed_max_devices_per_node,
    8,
    "Distributed training. The maximum number of devices per training node.");
DEFINE_string(
    distributed_rndv_filepath,
    "",
    "Distributed training. Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");

/* RUN OPTIONS */
DEFINE_string(
    exp_rundir,
    "",
    "Experiment path, where logs and models will be stored.");
DEFINE_string(
    exp_model_name,
    "model",
    "Name used to save model bin and model log. '--exp_rundir' will be used as prefix.");
DEFINE_string(
    exp_init_model_path,
    "",
    "Initialization model full path, used as init model to start training.");
DEFINE_double(
    exp_pct_train_eval,
    1,
    "[train] Percentage of training set (by number of utts) to use for evaluation");

/* DATA OPTIONS */
DEFINE_string(
    data_asr_dir,
    "",
    "Prefix for the 'data_train' and 'data_valid' files.");
DEFINE_string(
    data_asr_train,
    "",
    "Comma-separated list of training data files; '--data_dir' will be used to add prefix for the files.");
DEFINE_string(
    data_asr_valid,
    "",
    "Comma-separated list of validation/test data files; \
    '--data_dir' will be used to add prefix for the files.");
DEFINE_int64(
    data_asr_batch_size,
    1,
    "Batch size of data (per process in distributed training). \
    If '--data_use_dynamic_batching=true' is used can be different \
    to have '--data_tokens_per_sample' * '--data_batch_size' tokens in the batch.");
DEFINE_bool(
    data_asr_usewordpiece,
    false,
    "Specify if a word separator can be used inside of a token. "
    "Should be used if the SentencePiece tool is used to "
    "construct a token set containing word-pieces");
DEFINE_int64(
    data_asr_replabel,
    0,
    "Replace up to replabel reptitions by additional token classes");
DEFINE_string(
    data_asr_surround,
    "",
    "Surround target tokens sequence with provided token (duplicates are removed)");
DEFINE_bool(
    data_asr_eostoken,
    false,
    "Add the eos (end of sentence) token into the token set and append target token sequences with it "
    "at train, test, and decode time.");
DEFINE_string(
    data_asr_wordseparator,
    fl::app::asr::kSilToken,
    "Defines a word separator token used to map tokens sequences to words. "
    "Defaults to a pre-defined value.");
DEFINE_double(
    data_asr_sampletarget,
    0.0,
    "The probability [0.0, 1.0] with which targets are randomly sampled from a "
    "lexicon if multiple token constructions exist for a given word");

DEFINE_string(
    data_lm_dir,
    "",
    "Prefix for the 'data_train' and 'data_valid' files.");
DEFINE_string(
    data_lm_train,
    "",
    "Comma-separated list of training data files; '--data_dir' will be used to add prefix for the files.");
DEFINE_string(
    data_lm_valid,
    "",
    "Comma-separated list of validation/test data files; \
    '--data_dir' will be used to add prefix for the files.");
DEFINE_int64(
    data_lm_batch_size,
    1,
    "Batch size of data (per process in distributed training). \
    If '--data_use_dynamic_batching=true' is used can be different \
    to have '--data_tokens_per_sample' * '--data_batch_size' tokens in the batch.");
DEFINE_int64(
    data_lm_tokens_per_sample,
    1024,
    "Max number of tokens per sample in the data. \
    See details for '--data_sample_break_mode' and '--data_use_dynamic_batching'.");
DEFINE_string(
    data_lm_sample_break_mode,
    "none",
    "How to split sentences to form samples and batch. \
    'none' means split joined text into chunks of '--data_tokens_per_sample' tokens. \
    'eos' means split text by the end of sentence to create a sample, \
    if sentence len greater than '--data_tokens_per_sample' then exclude this sentence.");
DEFINE_bool(
    data_lm_use_dynamic_batching,
    false,
    "if or not use dynamic batching in case of '--data_sample_break_mode=eos'.");

DEFINE_int64(
    data_prefetch_threads,
    1,
    "[train] Number of threads for data parallelization (prefetching the data)");

/* DICTIONARY OPTIONS */
DEFINE_string(
    dictionary,
    "",
    "Path to the dictionary file (read/write), which defines tokens set of language model.");
DEFINE_int64(
    dictionary_max_size,
    -1,
    "Number of rows to use from the dictionary file (top rows), cutting the number of target classes.");
DEFINE_string(
    dictionary_tokens,
    "tokens.txt",
    "Tokens file path, the 'tokensdir' flag is used as a prefix for this path");
DEFINE_bool(dictionary_wordlm, true, "trianing a word LM?");

/* TRAIN OPTIONS */
DEFINE_string(train_task, "autoreg", "Task for training: autoreg or mask");
DEFINE_string(
    train_arch_dir,
    "",
    "Prefix for the arch file of a model description.");
DEFINE_string(
    train_arch_file,
    "model.arch",
    "Arch file path for the model description. '--train_arch_dir' is used as prefix for this path.");
DEFINE_string(
    train_asr_frontend_arch_file,
    "model.arch",
    "Arch file path for the asr frontend description. '--train_arch_dir' is used as prefix for this path.");
DEFINE_string(
    train_lm_frontend_arch_file,
    "model.arch",
    "Arch file path for the lm frontend description. '--train_arch_dir' is used as prefix for this path.");
DEFINE_int64(
    train_seed,
    0,
    "Manually specify Arrayfire seed for reproducibility.");
DEFINE_string(
    train_optimizer,
    "nag",
    "Optimizer to use in training. Supported for now: adagrad, sgd, nag.");

DEFINE_int64(
    train_warmup_updates,
    0,
    "Use warmup. Ramp learning rate from '--train_warmup_init_lr' till '--train_lr' \
    linearly during '--train_warmup_updates' number of updates.");
DEFINE_double(
    train_warmup_init_lr,
    0.0,
    "Warmup init learning rate from which ramping starts.");

DEFINE_double(
    train_lr,
    1.0,
    "Learning rate for optimization process. \
    If '--train_warmup' is used then lr is warmuped to '--train_lr' value.");
DEFINE_string(
    train_lr_schedule,
    "fixed",
    "Learning rate schedule. \
    'fixed' means 'train_lr' will be used all the time (except warmup stage); \
    'invsqrt' means decaying with respect to q/sqrt(nUpdates).");
DEFINE_double(
    train_momentum,
    0.0,
    "Momentum factor used in optimization process.");
DEFINE_double(
    train_weight_decay,
    0.0,
    "L2 penalty coefficient for the parameters during optimization process.");

DEFINE_double(
    train_max_grad_norm,
    0.0,
    "Clip gradients at this value (0 = no clipping).");
DEFINE_int64(
    train_save_updates,
    0,
    "Specifies to save model every '--train_save_updates' updates.");
DEFINE_int64(
    train_report_updates,
    0,
    "Number of updates after which we will run evaluation and save model, \
    if 0 we only do this at the end of epoch ");
DEFINE_int64(
    train_total_updates,
    std::numeric_limits<int64_t>::max(),
    "Total number of updates.");

/* MASK OPTIONS */
DEFINE_double(mask_prob, 0.15, "[mask lm task] Probability of masking.");
DEFINE_double(
    mask_rand_token_prob,
    0.1,
    "[mask lm task] Probability of mask token to set random token.");
DEFINE_double(
    mask_same_token_prob,
    0.1,
    "[mask lm task] Probability of mask token to set original token.");
DEFINE_int64(
    mask_min_length,
    1,
    "[mask lm task] Min number of masked tokens in each sample.");

/* NORMALIZATION OPTIONS */
DEFINE_int64(
    norm_localnrmlleftctx,
    0,
    "Left context size for local normalization of input "
    "audio after featurization (computation of MFCC, etc.)");
DEFINE_int64(
    norm_localnrmlrightctx,
    0,
    "Right context size for local normalization of input "
    "audio after featurization (computation of MFCC, etc.)");
DEFINE_string(
    norm_onorm,
    "none",
    "[train] Criterion normalization mode. One of: "
    "{'none' - no normalization, 'target' - by target size, "
    "'input' - by input size}");
DEFINE_bool(
    norm_sqnorm,
    false,
    "[train] Use square-root while normalizing criterion loss with 'onorm' mode");

/* FEATURE OPTIONS */
DEFINE_int64(
    feat_samplerate,
    16000,
    "Sample rate (Hz) for training, validation and test audio data");
DEFINE_bool(
    feat_mfcc,
    false,
    "Use standard htk mfcc features as input by processing audio "
    "(if 'mfcc', 'pow', 'mfsc' all false raw wave will be used as input)");
DEFINE_int64(feat_mfcccoeffs, 13, "Number of mfcc coefficients");
DEFINE_bool(
    feat_pow,
    false,
    "Use standard power spectrum as input by processing audio "
    "(if 'mfcc', 'pow', 'mfsc' all false raw wave will be used as input)");
DEFINE_bool(
    feat_mfsc,
    false,
    "Use standard mfsc features as input "
    "(if 'mfcc', 'pow', 'mfsc' all false raw wave will be used as input)");
DEFINE_double(
    feat_melfloor,
    1.0,
    "Specify optional mel floor for mfcc/mfsc/pow");
DEFINE_int64(
    feat_filterbanks,
    80,
    "Number of mel-filter bank channels, "
    "used also with RawSpecAugment to define number of mel-scale bins");
DEFINE_int64(
    feat_devwin,
    0,
    "Window length for delta and doubledelta derivatives");
DEFINE_int64(
    feat_fftcachesize,
    1,
    "Number of cached cuFFT plans in GPU memory");
DEFINE_int64(
    feat_framesizems,
    25,
    "Window size in millisecond for power spectrum features");
DEFINE_int64(
    feat_framestridems,
    10,
    "Stride in milliseconds for power spectrum features");
DEFINE_int64(
    feat_lowfreqfilterbank,
    0,
    "Low freq filter bank (Hz). "
    "Is used also in RawSpecAugment to define the lowest frequecny bound for augmentation");
DEFINE_int64(
    feat_highfreqfilterbank,
    -1,
    "High freq filter bank (Hz). "
    "Is used also in RawSpecAugment to define the highest frequecny bound for augmentation");

/* SPECAUGMENT OPTIONS */
DEFINE_int64(
    specaug_fmaskf,
    27,
    "[train] Maximum number of frequency bands / mel-scale bands "
    "that are masked in SpecAugment/RawSpecAugment");
DEFINE_int64(
    specaug_fmaskn,
    2,
    "[train] Number of frequency masks in SpecAugment/RawSpecAugment");
DEFINE_int64(
    specaug_tmaskt,
    100,
    "[train] Maximum number of frames (input elements) that are masked in SpecAugment/RawSpecAugment");
DEFINE_double(
    specaug_tmaskp,
    1.0,
    "[train] Maximum proportion of the input sequence (1.0 is 100%) "
    "that can be masked in time for SpecAugment/RawSpecAugment");
DEFINE_int64(
    specaug_tmaskn,
    2,
    "[train] Number of time masks in SpecAugment/RawSpecAugment");
DEFINE_int64(
    specaug_start_update,
    -1,
    "[train] Use SpecAugment starting at the update number inputted. -1 means no SpecAugment. "
    "In case of raw wav input ('mfcc', 'pow' and 'mfsc' are all false) "
    "we apply RawSpecAugment which emulates behaviour of SpecAugment");

/* SOUND EFFECT AUGMENTATION OPTIONS */
DEFINE_string(
    ssfx_config,
    "",
    "[train] Path to a sound effect json config file. When set the sound effect is "
    "applied to augment the input data.");

// MIXED PRECISION OPTIONS

DEFINE_string(
    amp_optim_mode,
    "",
    "[train] Sets the flashlight optimization mode. "
    "Optim modes can be O1, O2, or O3.");
DEFINE_bool(
    amp_use_mixed_precision,
    false,
    "[train] Use mixed precision for training - scale loss and gradients up and down "
    "by a scale factor that changes over time. If no fl optim mode is "
    "specified with --fl_optim_mode when passing this flag, automatically "
    "sets the optim mode to O1.");
DEFINE_double(
    amp_scale_factor,
    4096.,
    "[train] Starting scale factor to use for loss scaling "
    " with mixed precision training");
DEFINE_uint64(
    amp_scale_factor_update_interval,
    2000,
    "[train] Update interval for adjusting loss scaling in mixed precision training");
DEFINE_uint64(
    amp_max_scale_factor,
    32000,
    "[train] Maximum value for the loss scale factor in mixed precision training");

} // namespace joint
} // namespace app
} // namespace fl