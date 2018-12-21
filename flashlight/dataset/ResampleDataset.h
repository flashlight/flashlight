/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include "Dataset.h"

namespace fl {

/**
 * A view into a dataset, with indices permuted.
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
   * Constructs a ResampleDataset with the identity permutation.
   * @param[in] dataset The underlying dataset.
   */
  explicit ResampleDataset(std::shared_ptr<const Dataset> dataset);

  /**
   * Constructs a ResampleDataset with permutation specified by a vector.
   * @param[in] dataset The underlying dataset.
   * @param[in] resamplevec The vector specifying the permutation.
   */
  ResampleDataset(
      std::shared_ptr<const Dataset> dataset,
      std::vector<int64_t> resamplevec);

  /**
   * Constructs a ResampleDataset with permutation specified by a function.
   * The function should be deterministic.
   * @param[in] dataset The underlying dataset.
   * @param[in] resamplefn The function specifying the permutation.
   */
  ResampleDataset(
      std::shared_ptr<const Dataset> dataset,
      const PermutationFunction& resamplefn);

  virtual int64_t size() const override;

  virtual std::vector<af::array> get(const int64_t idx) const override;

  /**
   * Changes the permutation used to resample the dataset.
   * @param[in] resamplevec The vector specifying the new permutation.
   */
  void resample(std::vector<int64_t> resamplevec);

 protected:
  std::shared_ptr<const Dataset> dataset_;
  std::vector<int64_t> resampleVec_;
};
} // namespace fl
