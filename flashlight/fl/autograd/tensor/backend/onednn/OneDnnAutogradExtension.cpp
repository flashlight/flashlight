/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/backend/onednn/OneDnnAutogradExtension.h"

namespace fl {

bool OneDnnAutogradExtension::isDataTypeSupported(
    const fl::dtype& dtype) const {
  // fp16 computation is not supported with onednn
  return dtype != fl::dtype::f16;
}

} // namespace fl
