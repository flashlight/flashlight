/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"

namespace fl {

ScalarNode::ScalarNode(
    const Shape& shape,
    const fl::dtype type,
    const ScalarType scalar,
    PrivateHelper)
    : NodeTrait({}, shape), dtype_(type), scalar_(scalar) {}

dtype ScalarNode::dataType() const {
  return dtype_;
}

} // namespace fl
