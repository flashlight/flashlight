/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>
#include <flashlight/fl/dataset/Sample.h>

namespace fl {
namespace ext {
namespace image {

void loadJpeg(
    const std::string& fp,
    fl::SamplePtr samplePtr,
    int desiredNumberOfChannels = 3);

} // namespace image
} // namespace ext
} // namespace fl
