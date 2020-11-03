/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/flags/DecodeFlags.h"
#include <limits>

namespace fl {
namespace app {
namespace asr {

// DATA OPTIONS
DEFINE_string(test, "", "comma-separated list of test data");

// DECODER OPTIONS

DEFINE_bool(show, false, "show predictions");
DEFINE_bool(showletters, false, "show letter predictions");
DEFINE_bool(logadd, false, "use logadd when merging decoder nodes");
DEFINE_bool(uselexicon, true, "use lexicon in decoding");
DEFINE_bool(isbeamdump, false, "dump the decoding beam");

DEFINE_string(smearing, "none", "none, max or logadd");
DEFINE_string(lmtype, "kenlm", "kenlm, convlm");
DEFINE_string(lm_vocab, "", "path/to/lm_vocab.txt");
DEFINE_string(emission_dir, "", "path/to/emission_dir/");
DEFINE_string(lm, "", "path/to/language_model");
DEFINE_string(am, "", "path/to/acoustic_model");
DEFINE_string(sclite, "", "path/to/sclite to be written");
DEFINE_string(decodertype, "wrd", "wrd, tkn");

DEFINE_double(lmweight, 0.0, "language model weight");
DEFINE_double(wordscore, 0.0, "word insertion score");
DEFINE_double(silscore, 0.0, "silence insertion score");
DEFINE_double(
    unkscore,
    -std::numeric_limits<float>::infinity(),
    "unknown word insertion score");
DEFINE_double(eosscore, 0.0, "EOS insertion score");
DEFINE_double(beamthreshold, 25, "beam score threshold");

DEFINE_int32(maxload, -1, "max number of testing examples.");
DEFINE_int32(beamsize, 2500, "max overall beam size");
DEFINE_int32(beamsizetoken, 250000, "max beam for token selection");
DEFINE_int32(nthread_decoder_am_forward, 1, "number of threads for AM forward");
DEFINE_int32(nthread_decoder, 1, "number of threads for decoding");
DEFINE_int32(
    lm_memory,
    5000,
    "total memory size for batch during forward pass ");

DEFINE_int32(emission_queue_size, 3000, "max size of emission queue");
} // namespace asr
} // namespace app
} // namespace fl
