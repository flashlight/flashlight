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

DECLARE_bool(enable_distributed);
DECLARE_int64(world_rank);
DECLARE_int64(world_size);
DECLARE_int64(max_devices_per_node);
DECLARE_string(rndv_filepath);
} // namespace asr
} // namespace app
} // namespace fl
