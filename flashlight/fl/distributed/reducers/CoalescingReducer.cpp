/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/reducers/CoalescingReducer.h"
#include "flashlight/fl/distributed/DistributedApi.h"
#include "flashlight/fl/tensor/Compute.h"

namespace fl {

CoalescingReducer::CoalescingReducer(double scale, bool async, bool contiguous)
    : scale_(scale),
      async_(async),
      contiguous_(contiguous),
      cacheThresholdBytes_(DistributedConstants::kCoalesceCacheSize) {}

CoalescingReducer::~CoalescingReducer() {
  finalize();
}

void CoalescingReducer::add(Variable& var) {
  // if this tensor would push the cache oversize, flush
  if (currCacheSize_ + var.bytes() > cacheThresholdBytes_) {
    flush();
  }

  // check if the tensor is larger than the cache. If so, reduce immediately
  // and don't copy-coalesce
  if (var.bytes() > cacheThresholdBytes_) {
    allReduce(var, scale_, async_);
  } else {
    // if async, evaluating the JIT on the value upfront is more efficient than
    // evaluating the JIT for each Variable in the cache after we flush it,
    // since it more effectively facilitates overlapping compuation between the
    // AF and distributed compute streams.
    if (async_) {
      var.eval();
    }
    // otherwise, add to cache
    cache_.push_back(var);
    currCacheSize_ += var.bytes();
  }
}

void CoalescingReducer::finalize() {
  flush();
  synchronize();
}

void CoalescingReducer::flush() {
  allReduceMultiple(cache_, scale_, async_, contiguous_);
  currCacheSize_ = 0;
  cache_.clear();
}

void CoalescingReducer::synchronize() {
  if (async_ || contiguous_) {
    syncDistributed();
  }
}

} // namespace fl
