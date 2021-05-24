/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/contrib/contrib.h"

extern "C" fl::Module* createModule(int64_t nFeature, int64_t nLabel) {
  auto seq = new fl::Sequential();
  seq->add(std::make_shared<fl::Linear>(nFeature, nLabel));
  return seq;
}
