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
namespace asr {

DECLARE_int64(decoder_layers);
DECLARE_double(decoder_dropout);
DECLARE_double(decoder_layerdrop);

void initCompat();
} // namespace asr
} // namespace app
} // namespace fl
