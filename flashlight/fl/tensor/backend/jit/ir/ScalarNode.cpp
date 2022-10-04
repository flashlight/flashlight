/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"

namespace fl {

ScalarNode::ScalarNode(
    const Shape& shape,
    const fl::dtype type,
    const ScalarType scalar)
    : NodeTrait({}), shape_(shape), dtype_(type), scalar_(scalar) {}

const Shape& ScalarNode::shape() const {
  return shape_;
}

dtype ScalarNode::dataType() const {
  return dtype_;
}

} // namespace fl
