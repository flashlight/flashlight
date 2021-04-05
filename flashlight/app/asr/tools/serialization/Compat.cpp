/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/tools/serialization/Compat.h"

#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/common/Flags.h"

namespace fl {
namespace app {
namespace asr {
DEFINE_int64(decoder_layers, 1, "s2s transformer decoder: number of layers");
DEFINE_double(decoder_dropout, 0.0, "s2s transformer decoder: dropout");
DEFINE_double(decoder_layerdrop, 0.0, "s2s transformer decoder: layerdrop");

void initCompat() {
  DEPRECATE_FLAGS(decoder_layers, am_decoder_tr_layers);
  DEPRECATE_FLAGS(decoder_dropout, am_decoder_tr_dropout);
  DEPRECATE_FLAGS(decoder_layerdrop, am_decoder_tr_layerdrop);
}
} // namespace asr
} // namespace app
} // namespace fl
