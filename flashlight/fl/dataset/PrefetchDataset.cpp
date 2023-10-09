/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <stdexcept>

#include "flashlight/fl/common/Serialization.h"
#include "flashlight/fl/dataset/PrefetchDataset.h"
#include "flashlight/fl/tensor/Compute.h"

namespace fl {

PrefetchDataset::PrefetchDataset(
    std::shared_ptr<const Dataset> dataset,
    int64_t numThreads,
    int64_t prefetchSize)
    : dataset_(dataset),
      numThreads_(numThreads),
      prefetchSize_(prefetchSize),
      curIdx_(-1) {
  if (!dataset_) {
    throw std::invalid_argument("dataset to be prefetched is null");
  }
  if (!(numThreads_ > 0 && prefetchSize_ > 0) &&
      !(numThreads_ == 0 && prefetchSize_ == 0)) {
    throw std::invalid_argument("invalid numThreads or prefetchSize");
  }
  if (numThreads_ > 0) {
    auto deviceId = fl::getDevice();
    threadPool_ = std::make_unique<ThreadPool>(
        numThreads_,
        [deviceId](int /* threadId */) { fl::setDevice(deviceId); });
  }
}

std::vector<Tensor> PrefetchDataset::get(int64_t idx) const {
  checkIndexBounds(idx);

  if (numThreads_ == 0) {
    return dataset_->get(idx);
  }

  // remove from cache (if necessary)
  while (!prefetchCache_.empty() && idx != curIdx_) {
    prefetchCache_.pop();
    ++curIdx_;
  }

  // add to cache (if necessary)
  while (prefetchCache_.size() < prefetchSize_) {
    auto fetchIdx = idx + prefetchCache_.size();
    if (fetchIdx >= size()) {
      break;
    }
    prefetchCache_.emplace(threadPool_->enqueue(
        [this, fetchIdx]() { return this->dataset_->get(fetchIdx); }));
  }

  auto curSample = prefetchCache_.front().get();

  prefetchCache_.pop();
  curIdx_ = idx + 1;
  return curSample;
}

int64_t PrefetchDataset::size() const {
  return dataset_->size();
}
} // namespace fl
