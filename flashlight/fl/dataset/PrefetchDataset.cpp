/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
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
    int64_t prefetchSize,
    bool usePreallocatedSamples)
    : dataset_(dataset),
      numThreads_(numThreads),
      prefetchSize_(prefetchSize),
      usePreallocatedSamples_(usePreallocatedSamples),
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
  if (usePreallocatedSamples) {
    for (int i = 0; i < (prefetchSize_ * 2); i++) {
      data_.emplace_back(std::make_shared<Sample>());
      labels_.emplace_back(std::make_shared<Sample>());
    }
  }
}

std::vector<af::array> PrefetchDataset::get(int64_t idx) const {
  checkIndexBounds(idx);
  // Using the sample data structure is particularlay beneficial when it uses
  // non-pageable memory. However, acquiring and freeing this type of memory
  // is time consuming. To address this issue, the pinned memroy can be
  // preallocated during construction of samples, and samples should be reused.
  if (usePreallocatedSamples_) {
    // While there is room for loading a new training/validation example:
    // A. Get a `SamplePtr` for the data.
    // B. Get a `SamplePtr` for the label.
    // C. Launch a new thread to load the new example and saving the results in
    //    the aforementioned Sample Pointers.
    // D. The resulting future will be added to the `storageToSysmemQueue_`
    while (storageToSysmemQueue_.size() < prefetchSize_) {
      auto fetchIdx = idx + storageToSysmemQueue_.size();
      if (fetchIdx >= size()) {
        break;
      }
      auto dataPtr = data_.back();
      data_.pop_back();

      auto lblPtr = labels_.back();
      labels_.pop_back();
      storageToSysmemQueue_.emplace(
          threadPool_->enqueue([this, fetchIdx, dataPtr, lblPtr]() {
            std::vector<SamplePtr> ptrs = {dataPtr, lblPtr};
            this->dataset_->get(fetchIdx, ptrs);
            return ptrs;
          }));
    }

    do {
      // While a new element is ready in the `storageToSysmemQueue_` queue and
      // there is enough space in the `hostToDeviceQueue_`:
      // A. Pop the element from the former queue.
      // B. Call the `toDeviceAsync()` -- Non-blocking method.
      // C. Place the element in the `hostToDeviceQueue_`.
      while (!storageToSysmemQueue_.empty() &&
             hostToDeviceQueue_.size() < prefetchSize_ &&
             storageToSysmemQueue_.front().wait_for(std::chrono::seconds(0)) ==
                 std::future_status::ready) {
        auto batch = storageToSysmemQueue_.front().get();
        storageToSysmemQueue_.pop();
        for (int i = 0; i < batch.size(); i++) {
          batch[i]->toDeviceAsync();
        }
        hostToDeviceQueue_.push(batch);
      }
    } while (hostToDeviceQueue_.empty() && !storageToSysmemQueue_.empty());
    if (hostToDeviceQueue_.empty()) {
      throw std::runtime_error(
          "Trying to retrive samples from an empty queue.");
    }
    auto curSample = hostToDeviceQueue_.front();
    hostToDeviceQueue_.pop();

    auto result = {curSample[0]->array(), curSample[1]->array()};

    // Return the samples pointers to their corresponding vectors. Hence, they
    // can be reused in future. This eliminates the need for allocation of
    // new `pinned memory`.
    data_.push_back(curSample[0]);
    labels_.push_back(curSample[1]);
    return result;
  } else {
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
}

int64_t PrefetchDataset::size() const {
  return dataset_->size();
}
} // namespace fl
