/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/reducers/InlineReducer.h"
#include "flashlight/fl/distributed/DistributedApi.h"

namespace fl {

InlineReducer::InlineReducer(double scale) : scale_(scale) {}

void InlineReducer::add(Variable& var) {
  if (getWorldSize() > 1) {
    allReduce(var.tensor());
  }
  var.tensor() *= scale_;
}

} // namespace fl
