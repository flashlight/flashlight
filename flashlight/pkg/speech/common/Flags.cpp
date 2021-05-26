/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/common/Flags.h"
#include "flashlight/pkg/speech/common/Defines.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

namespace fl {
namespace pkg {
namespace speech {

// Flags that are specific to an executable are marked
// in the description with the executable usage in square parenthesis,
// e.g. [test, decode]

// DATA OPTIONS
DEFINE_string(
    train,
    "",
    "[train] Comma-separated list of training files where each row specifies sample "
    "information in the format [sample_id audio_absolute_path size transcription]");
DEFINE_string(
    valid,
    "",
    "[train] Comma-separated list of validation files where each row specifies sample "
    "information in the format [sample_id audio_absolute_path size transcription]");
DEFINE_string(
    test,
    "",
    "[test, decode] Comma-separated list of test files where each row specifies sample "
    "information in the format [sample_id audio_absolute_path size transcription]");
DEFINE_int64(
    batchsize,
    1,
    "[train] Batch size for training data (per process in distributed training)");
DEFINE_int64(
    validbatchsize,
    -1,
    "[train] Batch size for validation data (per process in distributed training). "
    "If -1 then use the value of the 'batchsize' flag");
DEFINE_int64(
    samplerate,
    16000,
    "Sample rate (Hz) for training, validation and test audio data");
DEFINE_int64(
    channels,
    1,
    "Number of input channels in training, validation and test audio data");
DEFINE_string(
    tokens,
    "tokens.txt",
    "Tokens file path, the 'tokensdir' flag is used as a prefix for this path");
DEFINE_string(
    batching_strategy,
    "none",
    "Batching strategy to use, supports {'none', 'dynamic', 'rand', 'randdynamic'}. "
    "When using 'none' strategy then batches of size 'batchsize' are created. "
    "When using 'dynamic' batching for training, 'batchsize' will be ignored "
    "and 'max_tokens' will be used to compute the effective batch size. "
    "To use unordered input data to pack batches, use either 'rand' "
    "or 'randdynamic' which shuffles data before packing, "
    " then follows the same packing strategies as 'none' or 'dynamic', respectively.");
DEFINE_int64(
    batching_max_duration,
    0,
    "Maximum number of tokens/frames in the batch when using 'dynamic' batching strategy. "
    "Measured with the same unit as input sizes are specified in data list files");
DEFINE_bool(
    usewordpiece,
    false,
    "Specify if a word separator can be used inside of a token. "
    "Should be used if the SentencePiece tool is used to "
    "construct a token set containing word-pieces");
DEFINE_int64(
    replabel,
    0,
    "Replace up to replabel reptitions by additional token classes");
DEFINE_string(
    surround,
    "",
    "Surround target tokens sequence with provided token (duplicates are removed)");
DEFINE_string(
    wordseparator,
    kSilToken,
    "Defines a word separator token used to map tokens sequences to words. "
    "Defaults to a pre-defined value.");
DEFINE_double(
    sampletarget,
    0.0,
    "The probability [0.0, 1.0] with which targets are randomly sampled from a "
    "lexicon if multiple token constructions exist for a given word");

// NORMALIZATION OPTIONS
DEFINE_int64(
    localnrmlleftctx,
    0,
    "Left context size for local normalization of input "
    "audio after featurization (computation of MFCC, etc.)");
DEFINE_int64(
    localnrmlrightctx,
    0,
    "Right context size for local normalization of input "
    "audio after featurization (computation of MFCC, etc.)");
DEFINE_string(
    onorm,
    "none",
    "[train] Criterion normalization mode. One of: "
    "{'none' - no normalization, 'target' - by target size, "
    "'input' - by input size}");
DEFINE_bool(
    sqnorm,
    false,
    "[train] Use square-root while normalizing criterion loss with 'onorm' mode");
DEFINE_bool(
    lrcosine,
    false,
    "[train] Use cosine learning rate schedule, see usage for more details");

// LEARNING HYPER-PARAMETER OPTIONS
DEFINE_int64(
    iter,
    std::numeric_limits<int64_t>::max(),
    "[train] Total number of updates for training");
DEFINE_bool(itersave, false, "Save model or not at each update");
DEFINE_double(lr, 1.0, "[train] Learning rate for the network parameters");
DEFINE_double(
    momentum,
    0.0,
    "[train] Momentum factor used in SGD optimizer for network only");
DEFINE_double(
    weightdecay,
    0.0,
    "[train] Weight decay (L2 penalty) for optimization for network only");
DEFINE_double(
    lrcrit,
    0,
    "[train] Criterion learning rate (for 'asg', 'seq2seq' and 'transformer' criterions)");
DEFINE_int64(
    warmup,
    1,
    "[train] Number of updates for warmup learning rate from 0 to 'lr'/'lrcrit' for network/criterion");
DEFINE_int64(
    saug_start_update,
    -1,
    "[train] Use SpecAugment starting at the update number inputted. -1 means no SpecAugment. "
    "In case of raw wav input ('mfcc', 'pow' and 'mfsc' are all false) "
    "we apply RawSpecAugment which emulates behaviour of SpecAugment");
DEFINE_int64(
    lr_decay,
    std::numeric_limits<int64_t>::max(),
    "[train] Epoch value when we start to decay 'lr'/'lrcrit'");
DEFINE_int64(
    lr_decay_step,
    std::numeric_limits<int64_t>::max(),
    "[train] Amount to decay 'lr' and 'lrcrit' starting from epoch 'lr_decay'");
DEFINE_double(
    maxgradnorm,
    0,
    "[train] Maximum gradient norm to which gradients exceeding it will be clipped (0 = no clipping)");
DEFINE_double(
    adambeta1,
    0.9,
    "[train] Beta1 parameter in the Adam, AMSGrad and NovoGrad optimizers");
DEFINE_double(
    adambeta2,
    0.999,
    "[train] Beta2 parameter in the Adam, AMSGrad and NovoGrad optimizers");
DEFINE_double(
    optimrho,
    0.9,
    "[train] Rho parameter in the RMSProp and Adadelta optimizers");
DEFINE_double(
    optimepsilon,
    1e-8,
    "[train] Epsilon parameter in the Adam, AMSGrad, NovoGrad, Adadelta, RMSProp and Adagrad optimizers");

// LR-SCHEDULER OPTIONS
DEFINE_int64(
    stepsize,
    std::numeric_limits<int64_t>::max(),
    "[train] Learning rate schedule if 'lrcosine=false'."
    "We multiply 'lr'/'lrcrit' by 'gamma' every 'stepsize' updates");
DEFINE_double(
    gamma,
    1.0,
    "[train] Learning rate annealing multiplier, see detail in 'stepsize' flag");

// OPTIMIZER OPTIONS
DEFINE_string(
    netoptim,
    kSGDOptimizer,
    "[train] Optimizer for the network, supported ones "
    "'sgd', 'adam', 'rmsprop', 'adadelta', 'adagrad', 'amsgrad', 'novograd'");
DEFINE_string(
    critoptim,
    kSGDOptimizer,
    "[train] Optimizer for the criterion (for 'asg', 'seq2seq' and 'transformer' criterions), "
    "supported ones 'sgd', 'adam', 'rmsprop', 'adadelta', 'adagrad', 'amsgrad', 'novograd'");

// MFCC OPTIONS
DEFINE_string(
    features_type,
    "mfsc",
    "Features type to compute input by processing audio. Could be "
    "mfcc: standard htk mfcc features; mfsc: standard mfsc features; "
    "pow: standard power spectrum; raw: raw wave");
DEFINE_int64(mfcccoeffs, 13, "Number of mfcc coefficients");
DEFINE_double(melfloor, 1.0, "Specify optional mel floor for mfcc/mfsc/pow");
DEFINE_int64(
    filterbanks,
    80,
    "Number of mel-filter bank channels, "
    "used also with RawSpecAugment to define number of mel-scale bins");
DEFINE_int64(devwin, 0, "Window length for delta and doubledelta derivatives");
DEFINE_int64(fftcachesize, 1, "Number of cached cuFFT plans in GPU memory");
DEFINE_int64(
    framesizems,
    25,
    "Window size in millisecond for power spectrum features");
DEFINE_int64(
    framestridems,
    10,
    "Stride in milliseconds for power spectrum features");
DEFINE_int64(
    lowfreqfilterbank,
    0,
    "Low freq filter bank (Hz). "
    "Is used also in RawSpecAugment to define the lowest frequecny bound for augmentation");
DEFINE_int64(
    highfreqfilterbank,
    -1,
    "High freq filter bank (Hz). "
    "Is used also in RawSpecAugment to define the highest frequecny bound for augmentation");

// SPECAUGMENT OPTIONS
DEFINE_int64(
    saug_fmaskf,
    27,
    "[train] Maximum number of frequency bands / mel-scale bands "
    "that are masked in SpecAugment/RawSpecAugment");
DEFINE_int64(
    saug_fmaskn,
    2,
    "[train] Number of frequency masks in SpecAugment/RawSpecAugment");
DEFINE_int64(
    saug_tmaskt,
    100,
    "[train] Maximum number of frames (input elements) that are masked in SpecAugment/RawSpecAugment");
DEFINE_double(
    saug_tmaskp,
    1.0,
    "[train] Maximum proportion of the input sequence (1.0 is 100%) "
    "that can be masked in time for SpecAugment/RawSpecAugment");
DEFINE_int64(
    saug_tmaskn,
    2,
    "[train] Number of time masks in SpecAugment/RawSpecAugment");

// SOUND EFFECTS AUGMENTATION OPTIONS
DEFINE_string(
    sfx_config,
    "",
    "[train] Path to a sound effect json config file. When set the sound effect is "
    "applied to augment the input data.");
DEFINE_int64(
    sfx_start_update,
    std::numeric_limits<int>::max(),
    "[train] Start sount effect augmentation starting at this update iteration.");

// RUN OPTIONS
DEFINE_string(datadir, "", "Prefix to the 'train'/'valid'/'test' files paths");
DEFINE_string(
    rundir,
    "",
    "[train] Name of the experiment root directory where logs, snapshots will be stored");
DEFINE_string(
    flagsfile,
    "",
    "File specifying gflags, could specify only part of flags");
DEFINE_int64(
    nthread,
    1,
    "[train] Number of threads for data parallelization (prefetching the data)");
DEFINE_int64(
    seed,
    0,
    "[train] Manually specify Arrayfire seed for reproducibility");
DEFINE_int64(
    reportiters,
    0,
    "[train] Number of updates after which we will run evaluation on validation data \
    and save model, if 0 we only do this at end of each epoch");
DEFINE_double(
    pcttraineval,
    100,
    "[train] Percentage of training set (by number of utts) to use for evaluation");
DEFINE_bool(
    fl_benchmark_mode,
    true,
    "[train] Sets flashlight benchmark mode, which dynamically "
    "benchmarks various operations based on their empirical performance on "
    "current hardware throughout training");
DEFINE_string(
    fl_optim_mode,
    "",
    "[train] Sets the flashlight optimization mode. "
    "Optim modes can be O1, O2, or O3.");
DEFINE_string(
    fl_log_level,
    "",
    "Sets the logging level - "
    "must be [FATAL, ERROR, WARNING, INFO]");
DEFINE_int64(fl_vlog_level, 0, "Sets the verbose logging level");

DEFINE_int64(
    fl_log_mem_ops_interval,
    0,
    "Flushes memory manager logs after a specified "
    "number of log entries. 1000000 is a reasonable "
    "value which will reduce overhead.");

// MIXED PRECISION OPTIONS
DEFINE_bool(
    fl_amp_use_mixed_precision,
    false,
    "[train] Use mixed precision for training - scale loss and gradients up and down "
    "by a scale factor that changes over time. If no fl optim mode is "
    "specified with --fl_optim_mode when passing this flag, automatically "
    "sets the optim mode to O1.");
DEFINE_double(
    fl_amp_scale_factor,
    4096.,
    "[train] Starting scale factor to use for loss scaling "
    " with mixed precision training");
DEFINE_uint64(
    fl_amp_scale_factor_update_interval,
    2000,
    "[train] Update interval for adjusting loss scaling in mixed precision training");
DEFINE_uint64(
    fl_amp_max_scale_factor,
    32000,
    "[train] Maximum value for the loss scale factor in mixed precision training");

// ARCHITECTURE OPTIONS
DEFINE_string(
    arch,
    "default",
    "[train] Network architecture file path");
DEFINE_string(
    criterion,
    kAsgCriterion,
    "[train] Training criterion to apply on top of network: 'asg', 'ctc', "
    "'seq2seq' (seq2seq with attention rnn decoder), "
    "'transformer' (seq2seq with attention and transfromer decoder)");
DEFINE_int64(
    encoderdim,
    0,
    "[train]: Dimension of encoded hidden state for 'seq2seq' and 'transformer' criterions");

// Seq2Seq Transformer decoder
DEFINE_int64(
    am_decoder_tr_layers,
    1,
    "[train]: 'transformer' criterion decoder architecture: number of layers");
DEFINE_double(
    am_decoder_tr_dropout,
    0.0,
    "[train]: 'transformer' criterion decoder architecture: dropout");
DEFINE_double(
    am_decoder_tr_layerdrop,
    0.0,
    "[train]: 'transformer' criterion decoder architecture: layerdrop");

// DECODER OPTIONS

DEFINE_bool(show, false, "[test, decode] Show predictions in the stdout");
DEFINE_bool(
    showletters,
    false,
    "[decode] Show tokens predictions in the stdout");
DEFINE_bool(
    logadd,
    false,
    "[decode] Use logadd operation when merging decoder nodes");
DEFINE_bool(
    uselexicon,
    true,
    "[test, decode] Use lexicon file to map between words and tokens sequence");
DEFINE_bool(isbeamdump, false, "[decode] Dump the decoding beam to the disk");

DEFINE_string(
    smearing,
    "none",
    "[decode] How to perform trie smearing to have proxy "
    "on scores in the middle of a word: 'none', 'max' or 'logadd'");
DEFINE_string(
    lmtype,
    "kenlm",
    "[decode] Language model type used along with acoustic model: 'kenlm', 'convlm'");
DEFINE_string(
    lexicon,
    "",
    "path/to/lexicon.txt which contains on each row space separated mapping of a word into tokens sequence");
DEFINE_string(
    lm_vocab,
    "",
    "[decode] path/to/lm_vocab.txt for the 'convlm' language model: each token is mapped to its file row index");
DEFINE_string(
    emission_dir,
    "",
    "path/to/emission_dir/ where emissions data will be stored");
DEFINE_string(lm, "", "[decode] path/to/language_model");
DEFINE_string(
    am,
    "",
    "path/to/acoustic_model, used also to continue and fork training");
DEFINE_string(sclite, "", "[decode] path/to/sclite to be written");
DEFINE_string(
    decodertype,
    "wrd",
    "[decode] Defines at which level language model should be applied: 'wrd', 'tkn'");

DEFINE_double(
    lmweight,
    0.0,
    "[decode] language model weight in the beam search");
DEFINE_double(
    wordscore,
    0.0,
    "[decode] word insertion score for lexicon-based decoding");
DEFINE_double(silscore, 0.0, "[decode] word separator insertion score");
DEFINE_double(
    unkscore,
    -std::numeric_limits<float>::infinity(),
    "[decode] unknown word insertion score");
DEFINE_double(
    eosscore,
    0.0,
    "[decode] End-of-sentence insertion score (for decoding of seq2seq with attention models)");
DEFINE_double(
    beamthreshold,
    25,
    "[decode] beam score threshold for early pruning of hypothesis");

DEFINE_int32(
    maxload,
    -1,
    "[test, decode] Maximum number of testing samples to process");
DEFINE_int32(
    maxword,
    -1,
    "Maximum number of words (rows) to use from the lexicon file");
DEFINE_int32(beamsize, 2500, "[decode] Maximum overall beam size");
DEFINE_int32(
    beamsizetoken,
    250000,
    "[decode] Maximum beam for tokens selection");
DEFINE_int32(
    nthread_decoder_am_forward,
    1,
    "[test, decoder] Number of threads for acoustic model forward");
DEFINE_int32(
    nthread_decoder,
    1,
    "[decode] Number of threads for beam-search decoding");
DEFINE_int32(
    lm_memory,
    5000,
    "[decode] Total memory size for batch forming for 'convlm' LM forward pass");

DEFINE_int32(
    emission_queue_size,
    3000,
    "[test, decode] Maximum size of emission queue for acoustic model forward pass");

DEFINE_double(
    smoothingtemperature,
    1.0,
    "[train] Smoothening the probability distribution in seq2seq "
    "decoder of 'seq2seq' and 'transformer' criterions");
DEFINE_int32(
    attentionthreshold,
    std::numeric_limits<int>::max(),
    "[train] Hard attention limit in seq2seq decoder only for 'seq2seq' criterion");

DEFINE_double(
    lmweight_low,
    0.0,
    "language model weight (low boundary, search)");
DEFINE_double(
    lmweight_high,
    4.0,
    "language model weight (high boundary, search)");
DEFINE_double(lmweight_step, 0.2, "language model weight (step, search)");

// ASG OPTIONS
DEFINE_int64(
    linseg,
    0,
    "[train] Number of updates of LinSeg to init transitions for ASG");
DEFINE_double(
    linlr,
    -1.0,
    "[train] LinSeg: learning rate for network parameters (if < 0, use lr)");
DEFINE_double(
    linlrcrit,
    -1.0,
    "[train] LinSeg criterion learning rate (if < 0, use lrcrit)");
DEFINE_double(
    transdiag,
    0.0,
    "[train] 'asg' criterion: initial value along diagonal of ASG transition matrix");

// SEQ2SEQ OPTIONS
DEFINE_int64(
    maxdecoderoutputlen,
    200,
    "'seq2seq'/'transformer' criterion: max decoder steps during inference; "
    "(for 'transformer' cannot be changed after initialization)");
DEFINE_int64(
    pctteacherforcing,
    100,
    "[train] 'seq2seq'/'transformer' criterion: percentage of steps to train using teacher forcing");
DEFINE_string(
    samplingstrategy,
    "rand",
    "[train] 'seq2seq'/'transformer' criterion: sampling strategy "
    "to use when `pctteacherforcing` < 100. One of: {'rand', 'model'}");
DEFINE_double(
    labelsmooth,
    0.0,
    "[train] 'seq2seq'/'transformer' criterion: fraction to smooth targets with uniform distribution.");
DEFINE_bool(
    inputfeeding,
    false,
    "[train] 'seq2seq' criterion: feed encoder summary to the decoder RNN");
DEFINE_string(
    attention,
    "content",
    "[train] 'seq2seq'/'transformer' criterion: attention type in the encoder-decoder, "
    "supported options: 'content', 'keyvalue', 'location', 'multi', 'multikv', 'multisplit', 'multikvsplit', "
    "'neural', 'neuralloc', 'simpleloc'");
DEFINE_string(
    attnWindow,
    "no",
    "[train] 'seq2seq'/'transformer' criterion: attention window type in the encoder-decoder, "
    "supported options: 'median', 'no', 'soft', 'softPretrain', 'step'");
DEFINE_int64(
    attndim,
    0,
    "[train] 'seq2seq'/'transformer' criterion: dimension of neural location attention");
DEFINE_int64(
    attnconvchannel,
    0,
    "[train] 'seq2seq'/'transformer' criterion: "
    "number of convolutional channels for location attention");
DEFINE_int64(
    attnconvkernel,
    0,
    "[train] 'seq2seq'/'transformer' criterion: kernel width for location attention");
DEFINE_int64(
    numattnhead,
    8,
    "[train] 'seq2seq'/'transformer' criterion: number of heads for multihead attention");
DEFINE_int64(
    leftWindowSize,
    50,
    "[train] 'seq2seq'/'transformer' criterion: left median window width");
DEFINE_int64(
    rightWindowSize,
    50,
    "[train] 'seq2seq'/'transformer' criterion: right median window width");
DEFINE_int64(
    maxsil,
    50,
    "[train] 'seq2seq'/'transformer' criterion: maximum number of "
    "leading silence frames for the step window");
DEFINE_int64(
    minsil,
    0,
    "[train] 'seq2seq'/'transformer' criterion: minimum number of "
    "leading silence frames for the step window");
DEFINE_double(
    maxrate,
    10,
    "[train] 'seq2seq'/'transformer' criterion: maximum ratio between the transcript "
    "and the encoded input lengths for the step window");
DEFINE_double(
    minrate,
    3,
    "[train] 'seq2seq'/'transformer' criterion: minimum ratio between the "
    "transcript and the encoded input lengths for the step window");
DEFINE_int64(
    softwoffset,
    10,
    "[train] 'seq2seq'/'transformer' criterion: offset for the soft "
    "window center (= offset + step * rate)");
DEFINE_double(
    softwrate,
    5,
    "[train] 'seq2seq'/'transformer' criterion: moving "
    "rate for the soft window center (= offset + step * rate)");
DEFINE_double(
    softwstd,
    5,
    "[train] 'seq2seq'/'transformer' criterion: std for the soft "
    "window shape (=exp(-(t - center)^2 / (2 * std^2)))");
DEFINE_bool(
    trainWithWindow,
    false,
    "[train] 'seq2seq'/'transformer' criterion: use "
    "force-aligned diagonal attention window during the whole training");
DEFINE_int64(
    pretrainWindow,
    0,
    "[train] 'seq2seq'/'transformer' criterion: use force-aligned diagonal attention window"
    "in training for 'pretrainWindow' updates");
DEFINE_double(
    gumbeltemperature,
    1.0,
    "[train] 'seq2seq' criterion decoder: temperature in gumbel softmax");
DEFINE_int64(
    decoderrnnlayer,
    1,
    "[train] 'seq2seq' criterion decoder: the number of decoder rnn layers");
DEFINE_int64(
    decoderattnround,
    1,
    "[train] 'seq2seq' criterion decoder: the number of decoder attention rounds");
DEFINE_double(
    decoderdropout,
    0.0,
    "[train] 'seq2seq' criterion decoder: dropout");

// DISTRIBUTED TRAINING
DEFINE_bool(enable_distributed, false, "[train] Enable distributed training");
DEFINE_int64(
    world_rank,
    0,
    "[train] Rank of the process (Used if rndv_filepath is not empty)");
DEFINE_int64(
    world_size,
    1,
    "[train] Total number of the processes (Used if rndv_filepath is not empty)");
DEFINE_int64(
    max_devices_per_node,
    8,
    "[train] The maximum number of devices per training node");
DEFINE_string(
    rndv_filepath,
    "",
    "[train] Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");

// FB SPECIFIC
DEFINE_bool(everstoredb, false, "use Everstore db for reading data");
DEFINE_bool(use_memcache, false, "use Memcache for reading data");

namespace detail {
/***************************** Deprecated Flags *****************************/
namespace {

void registerDeprecatedFlags() {
  // Register deprecated flags here using DEPRECATE_FLAGS. For example:
  // DEPRECATE_FLAGS(my_now_deprecated_flag_name, my_new_flag_name);
}

} // namespace

DeprecatedFlagsMap& getDeprecatedFlags() {
  static DeprecatedFlagsMap flagsMap = DeprecatedFlagsMap();
  return flagsMap;
}

void addDeprecatedFlag(
    const std::string& deprecatedFlagName,
    const std::string& newFlagName) {
  auto& map = getDeprecatedFlags();
  map.emplace(deprecatedFlagName, newFlagName);
}

bool isFlagSet(const std::string& name) {
  gflags::CommandLineFlagInfo flagInfo;
  if (!gflags::GetCommandLineFlagInfo(name.c_str(), &flagInfo)) {
    std::stringstream ss;
    ss << "Flag name " << name << " not found - check that it's declared.";
    throw std::invalid_argument(ss.str());
  }
  return !flagInfo.is_default;
}

} // namespace detail

void handleDeprecatedFlags() {
  auto& map = detail::getDeprecatedFlags();
  // Register flags
  static std::once_flag registerFlagsOnceFlag;
  std::call_once(registerFlagsOnceFlag, detail::registerDeprecatedFlags);

  for (auto& flagPair : map) {
    std::string deprecatedFlagValue;
    gflags::GetCommandLineOption(flagPair.first.c_str(), &deprecatedFlagValue);

    bool deprecatedFlagSet = detail::isFlagSet(flagPair.first);
    bool newFlagSet = detail::isFlagSet(flagPair.second);

    if (deprecatedFlagSet && newFlagSet) {
      // Use the new flag value
      std::cerr << "[WARNING] Both deprecated flag " << flagPair.first
                << " and new flag " << flagPair.second
                << " are set. Only the new flag will be "
                << "serialized when the model saved." << std::endl;
      ;
    } else if (deprecatedFlagSet && !newFlagSet) {
      std::cerr
          << "[WARNING] Usage of flag --" << flagPair.first
          << " is deprecated and has been replaced with "
          << "--" << flagPair.second
          << ". Setting the new flag equal to the value of the deprecated flag."
          << "The old flag will not be serialized when the model is saved."
          << std::endl;
      if (gflags::SetCommandLineOption(
              flagPair.second.c_str(), deprecatedFlagValue.c_str())
              .empty()) {
        std::stringstream ss;
        ss << "Failed to set new flag " << flagPair.second << " to value from "
           << flagPair.first << ".";
        throw std::logic_error(ss.str());
      }
    }

    // If the user set the new flag but not the deprecated flag, noop. If the
    // user set neither flag, noop.
  }
}
} // namespace speech
} // namespace pkg
} // namespace fl
