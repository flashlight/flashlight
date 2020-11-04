/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>

namespace fl {
namespace ext {
namespace image {

af::array loadJpeg(const std::string& fp, int desiredNumberOfChannels = 3);

} // namespace image
} // namespace ext
} // namespace fl
