/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/common/Defines.h"
#include <limits>


namespace fl {
namespace app {
namespace asr {

// DATA OPTIONS 
DEFINE_string(input, "flac", "input feature");
DEFINE_int64(samplerate, 16000, "sample rate (Hz)");
DEFINE_int64(channels, 1, "number of input channels");
DEFINE_string(tokens, "tokens.txt", "path/to/tokens");
DEFINE_bool(usewordpiece, false, "use word piece as target");
DEFINE_int64(
    replabel,
    0,
    "replace up to replabel reptitions by additional classes");
DEFINE_string(surround, "", "surround target with provided label");
DEFINE_bool(eostoken, false, "append target with end of sentence token");
DEFINE_string(
    dataorder,
    "input",
    "bin method to use for binning samples, input: in order of length of \
    input, input_spiral: binning using transcript(reference) length , \
    and spiral along audiolength, output_spiral: binning using audio length and \
    spiral along reference lenth");
DEFINE_int64(inputbinsize, 100, "Bin size along audio length axis");
DEFINE_int64(outputbinsize, 5, "Bin size along transcript length axis");
DEFINE_bool(blobdata, false, "use blobs instead of folders as input data");
DEFINE_string(
    wordseparator,
    kSilToken,
    "extra word boundaries to be inserted during target generation");

// FILTERING OPTIONS
DEFINE_int64(minisz, 0, "min input size (in msec) allowed during training");
DEFINE_int64(
    maxisz,
    std::numeric_limits<int64_t>::max(),
    "max input size (in msec) allowed during training");
DEFINE_int64(
    maxtsz,
    std::numeric_limits<int64_t>::max(),
    "max target size allowed during training");
DEFINE_int64(mintsz, 0, "min target size allowed during training");

// NORMALIZATION OPTIONS
DEFINE_int64(localnrmlleftctx, 0, "left context size for local normalization");
DEFINE_int64(
    localnrmlrightctx,
    0,
    "right context size for local normalization");
DEFINE_string(onorm, "none", "output norm (none");
DEFINE_bool(sqnorm, false, "use square-root while normalizing criterion loss");

// MFCC OPTIONS
DEFINE_bool(mfcc, false, "use standard htk mfcc features as input");
DEFINE_bool(pow, false, "use standard power spectrum as input");
DEFINE_int64(mfcccoeffs, 13, "number of mfcc coefficients");
DEFINE_bool(mfsc, false, "use standard mfsc features as input");
DEFINE_double(melfloor, 1.0, "specify optional mel floor for mfcc/mfsc/pow");
DEFINE_int64(filterbanks, 40, "Number of mel-filter bank channels");
DEFINE_int64(devwin, 0, "Window length for delta and doubledelta derivatives");
DEFINE_int64(fftcachesize, 1, "number of cached cuFFT plans in GPU memory");
DEFINE_int64(
    framesizems,
    25,
    "Window size in millisecond for power spectrum features");
DEFINE_int64(
    framestridems,
    10,
    "Stride millisecond for power spectrum feature");
DEFINE_int64(lowfreqfilterbank, 0, "low freq filter bank (Hz)");
DEFINE_int64(highfreqfilterbank, -1, "high freq filter bank (Hz)");

// RUN OPTIONS
DEFINE_string(datadir, "", "speech data directory");
DEFINE_string(tokensdir, "", "dictionary directory");
DEFINE_string(flagsfile, "", "File specifying gflags");
DEFINE_int64(seed, 0, "Manually specify Arrayfire seed.");
DEFINE_int64(
    memstepsize,
    10 * (1 << 20),
    "Minimum allocation size in bytes per array.");
DEFINE_string(lexicon, "", "path/to/lexicon.txt");
DEFINE_int32(maxword, -1, "maximum number of words to use");

// ARCHITECTURE OPTIONS
DEFINE_string(criterion, kAsgCriterion, "training criterion");

// SEQ2SEQ OPTIONS
DEFINE_int64(maxdecoderoutputlen, 200, "max decoder steps during inference");

// FB SPECIFIC
DEFINE_string(target, "tkn", "target feature");
DEFINE_bool(everstoredb, false, "use Everstore db for reading data");
DEFINE_bool(use_memcache, false, "use Memcache for reading data");
} // namespace asr
} // namespace app
} // namespace fl
