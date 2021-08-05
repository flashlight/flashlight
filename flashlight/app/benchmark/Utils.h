/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/app/benchmark/ModelBenchmarker.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

namespace fl {
namespace app {
namespace benchmark {

/**
 * Initialize training states and clear any existing history.
 */
void init();

/**
 * Log out the statistics of the current run. Details will also be logged out
 * when `verbose` is on.
 */
void printInfo(
    std::string&& name,
    bool fp16,
    const fl::app::benchmark::ModelBenchmarker& benchmarker,
    int numUnits,
    bool verbose = false);

} // namespace benchmark
} // namespace app
} // namespace fl
