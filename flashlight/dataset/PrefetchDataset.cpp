/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "PrefetchDataset.h"

namespace fl {

PrefetchDataset::PrefetchDataset(
    std::shared_ptr<const Dataset> dataset,
    int64_t numThreads,
    int64_t prefetchSize)
    : dataset_(dataset),
      numThreads_(numThreads),
      prefetchSize_(prefetchSize),
      curIdx_(-1) {
  FL_ASSERT(dataset_, "dataset cannot be null");
  FL_ASSERT(
      (numThreads_ > 0 && prefetchSize_ > 0) ||
          (numThreads_ == 0 && prefetchSize_ == 0),
      "Invalid value for numThreads/prefetchSize specified");
  if (numThreads_ > 0) {
    auto deviceId = af::getDevice();
    threadPool_ = std::unique_ptr<ThreadPool>(new ThreadPool(
        numThreads_,
        [deviceId](int /* threadId */) { af::setDevice(deviceId); }));
  }
}

std::vector<af::array> PrefetchDataset::get(int64_t idx) const {
  FL_ASSERT(
      idx >= 0 && idx < size(),
      "Invalid value of idx. idx should be in [0, size())");
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
