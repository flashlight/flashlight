/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/app/asr/common/Defines.h"
#include <gflags/gflags.h>

namespace fl {
namespace app {
namespace asr {

/* ========== DATA OPTIONS ========== */

DECLARE_string(input);
DECLARE_int64(samplerate);
DECLARE_int64(channels);
DECLARE_string(tokens);
DECLARE_bool(usewordpiece);
DECLARE_int64(replabel);
DECLARE_string(surround);
DECLARE_bool(eostoken);
DECLARE_string(dataorder);
DECLARE_int64(inputbinsize);
DECLARE_int64(outputbinsize);
DECLARE_bool(blobdata);
DECLARE_string(wordseparator);

/* ========== FILTERING OPTIONS ========== */

DECLARE_int64(minisz);
DECLARE_int64(maxisz);
DECLARE_int64(mintsz);
DECLARE_int64(maxtsz);

/* ========== NORMALIZATION OPTIONS ========== */

DECLARE_int64(localnrmlleftctx);
DECLARE_int64(localnrmlrightctx);
DECLARE_string(onorm);
DECLARE_bool(sqnorm);

/* ========== MFCC OPTIONS ========== */

DECLARE_bool(mfcc);
DECLARE_bool(pow);
DECLARE_int64(mfcccoeffs);
DECLARE_bool(mfsc);
DECLARE_double(melfloor);
DECLARE_int64(filterbanks);
DECLARE_int64(devwin);
DECLARE_int64(fftcachesize);
DECLARE_int64(framesizems);
DECLARE_int64(framestridems);
DECLARE_int64(lowfreqfilterbank);
DECLARE_int64(highfreqfilterbank);

/* ========== RUN OPTIONS ========== */

DECLARE_string(datadir);
DECLARE_string(tokensdir);
DECLARE_string(flagsfile);
DECLARE_int64(seed);
DECLARE_int64(memstepsize);
DECLARE_string(lexicon);
DECLARE_int32(maxword);

/* ========== ARCHITECTURE OPTIONS ========== */

DECLARE_string(criterion);

/* ========== SEQ2SEQ OPTIONS ========== */

DECLARE_int64(maxdecoderoutputlen);

/* ========== FB SPECIFIC ========== */
DECLARE_string(target);
DECLARE_bool(everstoredb);
DECLARE_bool(use_memcache);
} // namespace asr
} // namespace app
} // namespace fl
