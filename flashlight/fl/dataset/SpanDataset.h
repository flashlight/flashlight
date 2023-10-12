/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "flashlight/fl/dataset/Dataset.h"

namespace fl {

/**
 * A view into an underlying dataset with an offset and optional bounded length.
 *
 * The size of the `SpanDataset` is either specified for the size of the input
 dataset
 * accounting for the offset.
 *
 * We have, for example `SpanDataset(ds, 13).get(i) == ds.get(13 + i)`
 *
 * Example:
  \code{.cpp}
  // Make a datasets
  auto makeDataset = []() {
    auto tensor = fl::rand({5, 4, 10});
    std::vector<Tensor> fields{tensor};
    return std::make_shared<TensorDataset>(fields);
  };
  auto ds = makeDataset();

  // Create two spanned datasets
  SpanDataset spands1(ds, 2);
  SpanDataset spands2(ds, 0, 2);
  std::cout << spands1.size() << "\n"; // 8
  std::cout << spands2.size() << "\n"; // 2
  std::cout << allClose(spands1.get(3)[0], ds->get(5)[0]) << "\n"; // 1
  std::cout << allClose(spands2.get(1)[1], ds->get(1)[0]) << "\n"; // 1
  \endcode
 */
class FL_API SpanDataset : public Dataset {
 public:
  /**
   * Creates a SpanDataset.
   * @param[in] dataset The underlying dataset.
   * @param[in] offset The starting index of the new dataset relative to the
   * underlying dataset.
   * @param[in] length The size of the new dataset (if -1, uses previous size
   * minus the offset)
   */
  explicit SpanDataset(
      std::shared_ptr<const Dataset> dataset,
      const int64_t offset,
      const int64_t length = -1);

  int64_t size() const override;

  std::vector<Tensor> get(const int64_t idx) const override;

 private:
  std::shared_ptr<const Dataset> dataset_;
  int64_t offset_;
  int64_t size_;
};
} // namespace fl
