/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/DistributedApi.h"

#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace fl {

void distributedInit(
    DistributedInit /* initMethod */,
    int worldRank,
    int worldSize,
    const std::unordered_map<std::string, std::string>& /* params = {} */) {
  if (isDistributedInit()) {
    std::cerr << "warning: fl::distributedInit() called more than once\n";
    return;
  }
  if (worldSize > 1 || worldRank > 0) {
    throw std::runtime_error("worldSize must be 1 with distributed stub");
  }
  detail::DistributedInfo::getInstance().backend_ = DistributedBackend::STUB;
  detail::DistributedInfo::getInstance().isInitialized_ = true;
}

void allReduce(Tensor& arr, bool async /* = false */) {
  if (!isDistributedInit()) {
    throw std::runtime_error("distributed environment not initialized");
  }
  throw std::runtime_error("allReduce not supported for stub backend");
}

// Not yet supported
void allReduceMultiple(
    std::vector<Tensor*> arrs,
    bool async /* = false */,
    bool contiguous /* = false */) {
  throw std::runtime_error(
      "allReduceMultiple not supported for distributed stub backend");
}

void syncDistributed() {
  throw std::runtime_error(
      "Asynchronous allReduce not supported for distributed stub backend");
}

int getWorldRank() {
  return 0;
}

int getWorldSize() {
  return 1;
}
} // namespace fl
