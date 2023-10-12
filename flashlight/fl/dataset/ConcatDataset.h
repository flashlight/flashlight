/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/dataset/Dataset.h"

#include <vector>

namespace fl {

/**
 * A view into two or more underlying datasets with the indexes
 * concatenated in sequential order.
 *
 * Example:
  \code{.cpp}
  // Make two datasets with sizes 10 and 20
  auto makeDataset = [](int size) {
    auto tensor = fl::rand({5, 4, size});
    std::vector<Tensor> fields{tensor};
    return std::make_shared<TensorDataset>(fields);
  };
  auto ds1 = makeDataset(10);
  auto ds2 = makeDataset(20);

  // Concatenate them
  ConcatDataset concatds({ds1, ds2});
  std::cout << concatds.size() << "\n"; // 30
  std::cout << allClose(concatds.get(15)[0], ds2->get(5)[0]) << "\n"; // 1
  \endcode
 */
class FL_API ConcatDataset : public Dataset {
 public:
  /**
   * Creates a `ConcatDataset`.
   * @param[in] datasets The underlying datasets.
   */
  explicit ConcatDataset(
      const std::vector<std::shared_ptr<const Dataset>>& datasets);

  int64_t size() const override;

  std::vector<Tensor> get(const int64_t idx) const override;

 private:
  std::vector<std::shared_ptr<const Dataset>> datasets_;
  std::vector<int64_t> cumulativedatasetsizes_;
  int64_t size_;
};
} // namespace fl
