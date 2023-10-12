/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <future>
#include <queue>

#include "flashlight/fl/common/threadpool/ThreadPool.h"
#include "flashlight/fl/dataset/Dataset.h"

namespace fl {

/**
 * A view into a dataset, where a given number of samples are prefetched in
 * advance in a ThreadPool. PrefetchDataset should be used when there is a
 * sequential access to the underlying dataset. Otherwise, there will a lot of
 * cache misses leading to a degraded performance.
 *
 * Example:
  \code{.cpp}
  // Make a dataset with 100 samples
  auto tensor = fl::rand({5, 4, 100});
  std::vector<Tensor> fields{tensor};
  auto ds = std::make_shared<TensorDataset>(fields);

  // Iterate over the dataset using 4 background threads prefetching 2 samples
  // in advance
  for (auto& sample : PrefetchDataset(ds, 4, 2)) {
      // do something
  }
  \endcode
 */
class FL_API PrefetchDataset : public Dataset {
 public:
  /**
   * Creates a `PrefetchDataset`.
   * @param[in] dataset The underlying dataset.
   * @param[in] numThreads Number of threads used by the threadpool
   * @param[in] prefetchSize Number of samples to be prefetched
   */
  explicit PrefetchDataset(
      std::shared_ptr<const Dataset> dataset,
      int64_t numThreads,
      int64_t prefetchSize);

  int64_t size() const override;

  std::vector<Tensor> get(const int64_t idx) const override;

 protected:
  std::shared_ptr<const Dataset> dataset_;
  int64_t numThreads_, prefetchSize_;

 private:
  std::unique_ptr<ThreadPool> threadPool_;
  // state variables
  mutable std::queue<std::future<std::vector<Tensor>>> prefetchCache_;
  mutable int64_t curIdx_;
};

} // namespace fl
