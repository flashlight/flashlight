/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/flags/SharedFlags.h"

#include <gflags/gflags.h>

namespace fl {
namespace app {
namespace asr {

/* ========== DATA OPTIONS ========== */

DECLARE_string(test);

/* ========== DECODER OPTIONS ========== */

DECLARE_bool(show);
DECLARE_bool(showletters);
DECLARE_bool(logadd);
DECLARE_bool(uselexicon);
DECLARE_bool(isbeamdump);

DECLARE_string(smearing);
DECLARE_string(lmtype);
DECLARE_string(lm_vocab);
DECLARE_string(emission_dir);
DECLARE_string(lm);
DECLARE_string(am);
DECLARE_string(sclite);
DECLARE_string(decodertype);

DECLARE_double(lmweight);
DECLARE_double(wordscore);
DECLARE_double(silscore);
DECLARE_double(unkscore);
DECLARE_double(eosscore);
DECLARE_double(beamthreshold);

DECLARE_int32(maxload);
DECLARE_int32(beamsize);
DECLARE_int32(beamsizetoken);
DECLARE_int32(nthread_decoder_am_forward);
DECLARE_int32(nthread_decoder);
DECLARE_int32(lm_memory);

DECLARE_int32(emission_queue_size);
} // namespace asr
} // namespace app
} // namespace fl
