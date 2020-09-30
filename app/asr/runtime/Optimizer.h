/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/flashlight/flashlight.h"

#include "flashlight/app/asr/common/Defines.h"

namespace fl {
namespace app {
namespace asr {

std::shared_ptr<fl::FirstOrderOptimizer> initOptimizer(
    const std::vector<std::shared_ptr<fl::Module>>& nets,
    const std::string& optimizer,
    double lr,
    double momentum,
    double weightdecay);
}
} // namespace app
} // namespace fl
