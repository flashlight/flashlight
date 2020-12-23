/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "flashlight/fl/dataset/Dataset.h"
#include "flashlight/fl/dataset/Utils.h"

namespace fl {

/**
 * Policy for handling corner cases when the dataset size is not
 * exactly divisible by `batchsize` while performing batching.
 */
enum class BatchDatasetPolicy {
  /// The last samples not evenly divisible by `batchsize` are packed
  /// into a smaller-than-usual batch.
  INCLUDE_LAST = 0,
  /// The last samples not evenly divisible by `batchsize` are skipped.
  SKIP_LAST = 1,
  /// Constructor raises an error if sizes are not divisible.
  DIVISIBLE_ONLY = 2,
};
// TODO: add RANDOM_LAST to fill up last examples with random ones?

/**
 * A view into a dataset where samples are packed into batches.
 *
 * By default, for each field, the inputs must all have the same dimensions,
 * and it batches along the first singleton dimension.
 *
 * Example:
  \code{.cpp}
  // Make a dataset containing 42 tensors of dims [5, 4]
  auto tensor = af::randu(5, 4, 42);
  std::vector<af::array> fields{{tensor}};
  auto ds = std::make_shared<TensorDataset>(fields);

  // Batch them with batchsize=10
  BatchDataset batchds(ds, 10, BatchDatasetPolicy::INCLUDE_LAST);
  std::cout << batchds.get(0)[0].dims() << "\n"; // 5 4 10 1
  std::cout << batchds.get(4)[0].dims() << "\n"; // 5 4 2 1

  // create batch sizes vector specifying the each batch size (dynamic)
  std::vector<int64_t> batchSizes = {5, 10, 5, 10, 2, 10}

  // Batch them with batchSizes
  DynamicBatchDataset batchdsDynamic(ds, batchSizes);
  std::cout << batchdsDynamic.get(0)[0].dims() << "\n"; // 5 4 5 1
  std::cout << batchdsDynamic.get(5)[0].dims() << "\n"; // 5 4 10 1
  \endcode
 */
class BatchDataset : public Dataset {
 public:
  /**
   * Creates a `BatchDataset`.
   * @param[in] dataset The underlying dataset.
   * @param[in] batchsize The desired batch size.
   * @param[in] policy How to handle the last batch if sizes are indivisible.
   * @param[in] batchfns Custom batch function to use for difference indices.
   */
  BatchDataset(
      std::shared_ptr<const Dataset> dataset,
      int64_t batchsize,
      BatchDatasetPolicy policy = BatchDatasetPolicy::INCLUDE_LAST,
      const std::vector<BatchFunction>& batchfns = {});

  /**
   * Creates a `BatchDataset`.
   * @param[in] dataset The underlying dataset.
   * @param[in] batchSizes desired batch sizes (dynamic).
   * @param[in] batchfns Custom batch function to use for difference indices.
   */
  BatchDataset(
      std::shared_ptr<const Dataset> dataset,
      const std::vector<int64_t>& batchSizes,
      const std::vector<BatchFunction>& batchfns = {});

  int64_t size() const override;

  std::vector<af::array> get(const int64_t idx) const override;

 private:
  std::shared_ptr<const Dataset> dataset_;
  int64_t batchSize_;
  BatchDatasetPolicy batchPolicy_;
  std::vector<int64_t> cumSumBatchSize_;
  std::vector<BatchFunction> batchFns_;

  int64_t preBatchSize_; // Size of the dataset before batching
  int64_t size_;
};
} // namespace fl
