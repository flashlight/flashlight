/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/app/asr/common/Defines.h"

#include <gflags/gflags.h>
#include "flashlight/app/asr/flags/DistributedFlags.h"
#include "flashlight/app/asr/flags/SharedFlags.h"

namespace fl {
namespace app {
namespace asr {

/* ========== DATA OPTIONS ========== */

DECLARE_string(train);
DECLARE_string(valid);
DECLARE_int64(batchsize);
DECLARE_int64(validbatchsize);
DECLARE_bool(noresample);
DECLARE_double(sampletarget);

/* ========== LEARNING HYPER-PARAMETER OPTIONS ========== */

DECLARE_bool(lrcosine);
DECLARE_int64(iter);
DECLARE_bool(itersave);
DECLARE_double(lr);
DECLARE_double(momentum);
DECLARE_double(weightdecay);
DECLARE_double(lrcrit);
DECLARE_int64(warmup);
DECLARE_int64(saug_start_update);
DECLARE_int64(lr_decay);
DECLARE_int64(lr_decay_step);
DECLARE_double(maxgradnorm);
DECLARE_double(adambeta1); // TODO rename into optim beta1
DECLARE_double(adambeta2); // TODO rename into optim beta2
DECLARE_double(optimrho);
DECLARE_double(optimepsilon);

/* ========== LR-SCHEDULER OPTIONS ========== */

DECLARE_int64(stepsize);
DECLARE_double(gamma);

/* ========== OPTIMIZER OPTIONS ========== */
DECLARE_string(netoptim);
DECLARE_string(critoptim);

/* ========== SPECAUGMENT OPTIONS ========== */

DECLARE_int64(saug_fmaskf);
DECLARE_int64(saug_fmaskn);
DECLARE_int64(saug_tmaskt);
DECLARE_double(saug_tmaskp);
DECLARE_int64(saug_tmaskn);

/* ========== RUN OPTIONS ========== */

DECLARE_string(rundir);
DECLARE_string(archdir);
DECLARE_string(runname);
DECLARE_int64(nthread);
DECLARE_string(tag);
DECLARE_int64(reportiters);
DECLARE_double(pcttraineval);
DECLARE_bool(fl_benchmark_mode);
DECLARE_string(fl_optim_mode);

/* ========== MIXED PRECISION OPTIONS ========== */

DECLARE_bool(fl_amp_use_mixed_precision);
DECLARE_uint64(fl_amp_scale_factor);
DECLARE_uint64(fl_amp_scale_factor_update_interval);
DECLARE_uint64(fl_amp_max_scale_factor);

/* ========== ARCHITECTURE OPTIONS ========== */

DECLARE_string(arch);
DECLARE_int64(encoderdim);

// Seq2Seq Transformer decoder
DECLARE_int64(am_decoder_tr_layers);
DECLARE_double(am_decoder_tr_dropout);
DECLARE_double(am_decoder_tr_layerdrop);

DECLARE_string(lexicon);

// Seq2Seq
DECLARE_double(smoothingtemperature);
DECLARE_int32(attentionthreshold);

/* ========== ASG OPTIONS ========== */

DECLARE_int64(linseg);
DECLARE_double(linlr);
DECLARE_double(linlrcrit);
DECLARE_double(transdiag);

/* ========== SEQ2SEQ OPTIONS ========== */

DECLARE_int64(pctteacherforcing);
DECLARE_string(samplingstrategy);
DECLARE_double(labelsmooth);
DECLARE_bool(inputfeeding);
DECLARE_string(attention);
DECLARE_string(attnWindow);
DECLARE_int64(attndim);
DECLARE_int64(attnconvchannel);
DECLARE_int64(attnconvkernel);
DECLARE_int64(numattnhead);
DECLARE_int64(leftWindowSize);
DECLARE_int64(rightWindowSize);
DECLARE_int64(maxsil);
DECLARE_int64(minsil);
DECLARE_double(maxrate);
DECLARE_double(minrate);
DECLARE_int64(softwoffset);
DECLARE_double(softwrate);
DECLARE_double(softwstd);
DECLARE_bool(trainWithWindow);
DECLARE_int64(pretrainWindow);
DECLARE_double(gumbeltemperature);
DECLARE_int64(decoderrnnlayer);
DECLARE_int64(decoderattnround);
DECLARE_double(decoderdropout);
} // namespace asr
} // namespace app
} // namespace fl
