/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/contrib/contrib.h"
#include "flashlight/fl/flashlight.h"

namespace fl {
namespace pkg {
namespace speech {

using GetConvLmScoreFunc = std::function<std::vector<
    float>(const std::vector<int>&, const std::vector<int>&, int, int)>;

GetConvLmScoreFunc buildGetConvLmScoreFunction(std::shared_ptr<Module> network);
} // namespace speech
} // namespace pkg
} // namespace fl
