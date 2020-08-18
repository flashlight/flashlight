/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/contrib/contrib.h>
#include <flashlight/flashlight.h>

namespace fl {
namespace app {
namespace asr {

using GetConvLmScoreFunc = std::function<std::vector<
    float>(const std::vector<int>&, const std::vector<int>&, int, int)>;

GetConvLmScoreFunc buildGetConvLmScoreFunction(std::shared_ptr<Module> network);
} // namespace asr
} // namespace app
} // namespace fl
