/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/flags/DistributedFlags.h"

namespace fl {
namespace app {
namespace asr {

DEFINE_bool(enable_distributed, false, "enable distributed training");
DEFINE_int64(
    world_rank,
    0,
    "rank of the process (Used if rndv_filepath is not empty)");
DEFINE_int64(
    world_size,
    1,
    "total number of the process (Used if rndv_filepath is not empty)");
DEFINE_int64(
    max_devices_per_node,
    8,
    "the maximum number of devices per training node");
DEFINE_string(
    rndv_filepath,
    "",
    "Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");
} // namespace asr
} // namespace app
} // namespace fl
