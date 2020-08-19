/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "libraries/audio/criterion/Defines.h"

namespace fl {
namespace task {
namespace asr {

// sampling strategy to use in decoder in place of teacher forcing
constexpr const char* kModelSampling = "model";
constexpr const char* kRandSampling = "rand";
constexpr const char* kGumbelSampling = "gumbel";

} 
}
}
