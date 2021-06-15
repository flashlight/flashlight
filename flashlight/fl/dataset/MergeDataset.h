/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/dataset/Dataset.h"

#include <vector>

namespace fl {

/**
 * A view into two or more underlying datasets with the same indexes,
 * but with fields combined from all the datasets.
 *
 * The size of the `MergeDataset` is the max of the sizes of the input datasets.
 *
 * We have `MergeDataset({ds1, ds2}).get(i) == merge(ds1.get(i), ds2.get(i))`
 * where `merge` concatenates the `std::vector<af::array>` from each dataset.
 *
 * Example:
  \code{.cpp}
  // Make two datasets
  auto makeDataset = []() {
    auto tensor = af::randu(5, 4, 10);
    std::vector<af::array> fields{tensor};
    return std::make_shared<TensorDataset>(fields);
  };
  auto ds1 = makeDataset();
  auto ds2 = makeDataset();

  // Merge them
  MergeDataset mergeds({ds1, ds2});
  std::cout << mergeds.size() << "\n"; // 10
  std::cout << allClose(mergeds.get(5)[0], ds1->get(5)[0]) << "\n"; // 1
  std::cout << allClose(mergeds.get(5)[1], ds2->get(5)[0]) << "\n"; // 1
  \endcode
 */
class MergeDataset : public Dataset {
 public:
  /**
   * Creates a MergeDataset.
   * @param[in] datasets The underlying datasets.
   */
  explicit MergeDataset(
      const std::vector<std::shared_ptr<const Dataset>>& datasets);

  int64_t size() const override;

  std::vector<af::array> get(const int64_t idx) const override;

 private:
  std::vector<std::shared_ptr<const Dataset>> datasets_;
  int64_t size_;
};
} // namespace fl
