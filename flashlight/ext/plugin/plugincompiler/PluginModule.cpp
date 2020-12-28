/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>

#include "flashlight/fl/flashlight.h"

/**
 * This is a placeholder plugin module.
 *
 * Modifications SHOULD NOT be committed, but this file should be replaced with
 * valid plugin code when doing internal compilation with buck. In external
 * environments when building with CMake, this file is irrelevant as plugin
 * source paths can be freely specified.
 */
extern "C" fl::Module* createModule(int64_t nFeature, int64_t nLabel) {
  auto seq = std::make_unique<fl::Sequential>(); // placeholder
  return seq.release();
}
