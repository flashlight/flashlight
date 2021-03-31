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
inline const std::vector<float> kImageNetMean = {0.485, 0.456, 0.406};
inline const std::vector<float> kImageNetStd = {0.229, 0.224, 0.225};

} // namespace image
} // namespace app
} // namespace fl
