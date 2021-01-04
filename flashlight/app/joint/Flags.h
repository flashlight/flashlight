/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gflags/gflags.h>

namespace fl {
namespace app {
namespace joint {

/* CRITERION OPTIONS */
DECLARE_string(loss_type);
DECLARE_int64(loss_adsm_input_size);
DECLARE_string(loss_adsm_cutoffs);

/* DISTRIBUTED TRAINING */
DECLARE_bool(distributed_enable);
DECLARE_int64(distributed_world_rank);
DECLARE_int64(distributed_world_size);
DECLARE_int64(distributed_max_devices_per_node);
DECLARE_string(distributed_rndv_filepath);

/* RUN OPTIONS */
DECLARE_string(exp_rundir);
DECLARE_string(exp_model_name);
DECLARE_string(exp_init_model_path);
DECLARE_double(exp_pct_train_eval);

/* DATA OPTIONS */
DECLARE_string(data_asr_dir);
DECLARE_string(data_asr_train);
DECLARE_string(data_asr_valid);
DECLARE_int64(data_asr_batch_size);
DECLARE_bool(data_asr_usewordpiece);
DECLARE_int64(data_asr_replabel);
DECLARE_string(data_asr_surround);
DECLARE_bool(data_asr_eostoken);
DECLARE_string(data_asr_wordseparator);
DECLARE_double(data_asr_sampletarget);

DECLARE_string(data_lm_dir);
DECLARE_string(data_lm_train);
DECLARE_string(data_lm_valid);
DECLARE_int64(data_lm_batch_size);
DECLARE_int64(data_lm_tokens_per_sample);
DECLARE_string(data_lm_sample_break_mode);
DECLARE_bool(data_lm_use_dynamic_batching);

DECLARE_int64(data_prefetch_threads);


/* DICTIONARY OPTIONS */
DECLARE_string(dictionary);
DECLARE_int64(dictionary_max_size);
DECLARE_string(dictionary_tokens);


/* TRAIN OPTIONS */
DECLARE_string(train_task);
DECLARE_string(train_arch_dir);
DECLARE_string(train_arch_file);
DECLARE_string(train_asr_frontend_arch_file);
DECLARE_string(train_lm_frontend_arch_file);
DECLARE_int64(train_seed);
DECLARE_string(train_optimizer);
DECLARE_int64(train_warmup_updates);
DECLARE_double(train_warmup_init_lr);
DECLARE_double(train_lr);
DECLARE_string(train_lr_schedule);
DECLARE_double(train_momentum);
DECLARE_double(train_weight_decay);
DECLARE_double(train_max_grad_norm);
DECLARE_int64(train_save_updates);
DECLARE_int64(train_report_updates);
DECLARE_int64(train_total_updates);

/* MASK OPTIONS */
DECLARE_double(mask_prob);
DECLARE_double(mask_rand_token_prob);
DECLARE_double(mask_same_token_prob);
DECLARE_int64(mask_min_length);

/* NORMALIZATION OPTIONS */
DECLARE_int64(norm_localnrmlleftctx);
DECLARE_int64(norm_localnrmlrightctx);
DECLARE_string(norm_onorm);
DECLARE_bool(norm_sqnorm);

/* FEATURE OPTIONS */
DECLARE_int64(feat_samplerate);
DECLARE_bool(feat_mfcc);
DECLARE_bool(feat_pow);
DECLARE_int64(feat_mfcccoeffs);
DECLARE_bool(feat_mfsc);
DECLARE_double(feat_melfloor);
DECLARE_int64(feat_filterbanks);
DECLARE_int64(feat_devwin);
DECLARE_int64(feat_fftcachesize);
DECLARE_int64(feat_framesizems);
DECLARE_int64(feat_framestridems);
DECLARE_int64(feat_lowfreqfilterbank);
DECLARE_int64(feat_highfreqfilterbank);

/* SPECAUGMENT OPTIONS */
DECLARE_int64(specaug_fmaskf);
DECLARE_int64(specaug_fmaskn);
DECLARE_int64(specaug_tmaskt);
DECLARE_double(specaug_tmaskp);
DECLARE_int64(specaug_tmaskn);
DECLARE_int64(specaug_start_update);

/* SOUND EFFECT AUGMENTATION OPTIONS */
DECLARE_string(ssfx_config);

}
}
}