/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/dataset/datasets.h"

namespace fl {
namespace ext {
namespace image {

std::shared_ptr<Dataset> jpegLoader(std::vector<std::string> fps);

}
}
}
