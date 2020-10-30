/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/reducers/InlineReducer.h"
#include "flashlight/fl/distributed/DistributedApi.h"

namespace fl {

InlineReducer::InlineReducer(double scale) : scale_(scale) {}

void InlineReducer::add(Variable& var) {
  if (getWorldSize() > 1) {
    allReduce(var.array());
  }
  var.array() *= scale_;
}

} // namespace fl
