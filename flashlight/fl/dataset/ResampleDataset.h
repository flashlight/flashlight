/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include "flashlight/fl/dataset/Dataset.h"

namespace fl {

/**
 * A view into a dataset, with indices remapped.
 * Note: the mapping doesn't have to be bijective.
 *
 * Example:
  \code{.cpp}
  // Make a dataset with 10 samples
  auto tensor = af::randu(5, 4, 10);
  std::vector<af::array> fields{tensor};
  auto ds = std::make_shared<TensorDataset>(fields);

  // Resample it by reversing it
  auto permfn = [ds](int64_t x) { return ds->size() - 1 - x; };
  ResampleDataset resampleds(ds, permfn);
  std::cout << resampleds.size() << "\n"; // 10
  std::cout << allClose(resampleds.get(9)[0], ds->get(0)[0]) << "\n"; // 1
  \endcode
 */
class ResampleDataset : public Dataset {
 public:
  /**
   * Constructs a ResampleDataset with the identity mapping:
   * `ResampleDataset(ds)->get(i) == ds->get(i)`
   * @param[in] dataset The underlying dataset.
   */
  explicit ResampleDataset(std::shared_ptr<const Dataset> dataset);

  /**
   * Constructs a ResampleDataset with mapping specified by a vector:
   * `ResampleDataset(ds, v)->get(i) == ds->get(v[i])`
   * @param[in] dataset The underlying dataset.
   * @param[in] resamplevec The vector specifying the mapping.
   */
  ResampleDataset(
      std::shared_ptr<const Dataset> dataset,
      std::vector<int64_t> resamplevec);

  /**
   * Constructs a ResampleDataset with mapping specified by a function:
   * `ResampleDataset(ds, fn)->get(i) == ds->get(fn(i))`
   * The function should be deterministic.
   * @param[in] dataset The underlying dataset.
   * @param[in] resamplefn The function specifying the mapping.
   * @param[in] n The size of the new dataset (if -1, uses previous size)
   */
  ResampleDataset(
      std::shared_ptr<const Dataset> dataset,
      const PermutationFunction& resamplefn,
      int n = -1);

  int64_t size() const override;

  std::vector<af::array> get(const int64_t idx) const override;

  /**
   * Changes the mapping used to resample the dataset.
   * @param[in] resamplevec The vector specifying the new mapping.
   */
  void resample(std::vector<int64_t> resamplevec);

 protected:
  std::shared_ptr<const Dataset> dataset_;
  std::vector<int64_t> resampleVec_;
};
} // namespace fl
