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
namespace image {

constexpr int kNumImageNetClasses = 1000;
extern const std::vector<float> kImageNetMean;
extern const std::vector<float> kImageNetStd;

} // namespace image
} // namespace app
} // namespace fl
