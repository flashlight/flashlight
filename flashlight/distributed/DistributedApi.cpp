/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/distributed/DistributedApi.h"

namespace fl {

bool isDistributedInit() {
  return detail::DistributedInfo::getInstance().isInitialized_;
}

DistributedBackend distributedBackend() {
  return detail::DistributedInfo::getInstance().backend_;
}

void allReduce(
    Variable& var,
    double scale /* = 1.0 */,
    bool async /* = false */) {
  if (getWorldSize() > 1) {
    allReduce(var.array(), async);
  }
  var.array() *= scale;
}

void allReduceMultiple(
    std::vector<Variable> vars,
    double scale /* = 1.0 */,
    bool async /* = false */,
    bool contiguous /* = false */) {
  // return a vector of pointers to avoid copying
  std::vector<af::array*> arrs;
  for (auto& var : vars) {
    arrs.push_back(&var.array());
  }
  if (getWorldSize() > 1) {
    allReduceMultiple(arrs, async, contiguous);
  }
  for (auto& var : vars) {
    var.array() *= scale;
  }
}

namespace detail {
/*  static */ DistributedInfo& DistributedInfo::getInstance() {
  static DistributedInfo dinfo;
  return dinfo;
}
} // namespace detail

} // namespace fl
