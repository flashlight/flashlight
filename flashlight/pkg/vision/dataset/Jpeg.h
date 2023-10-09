/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/dataset/datasets.h"

namespace fl {
namespace pkg {
namespace vision {

Tensor loadJpeg(const std::string& fp, int desiredNumberOfChannels = 3);

std::shared_ptr<Dataset> jpegLoader(std::vector<std::string> fps);

} // namespace vision
} // namespace pkg
} // namespace fl
