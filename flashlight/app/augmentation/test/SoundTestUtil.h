/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

namespace fl {
namespace app {
namespace sfx {

std::vector<float>
genSinWave(size_t numSamples, size_t freq, size_t sampleRate, float amplitude);

} // namespace sfx
} // namespace app
} // namespace fl
