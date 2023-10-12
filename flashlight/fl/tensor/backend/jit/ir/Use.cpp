/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/Use.h"

namespace fl {

Use::Use(Node& user, unsigned inputIdx) : user_(user), inputIdx_(inputIdx) {}

unsigned Use::inputIdx() const {
  return inputIdx_;
}

Node& Use::user() const {
  return user_;
}

} // namespace fl
