/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/dataset/Dataset.h"

namespace fl {

/**
 * A view into a dataset with values transformed via the specified function(s).
 *
 * A different transformation may be specified for each array in the input.
 * A null TransformFunction specifies the identity transformation.
 * The dataset size remains unchanged.
 *
 * Example:
  \code{.cpp}
  // Make a dataset with 10 samples
  auto tensor = fl::rand({5, 4, 10});
  std::vector<Tensor> fields{tensor};
  auto ds = std::make_shared<TensorDataset>(fields);

  // Transform it
  auto negate = [](const Tensor& arr) { return -arr; };
  TransformDataset transformds(ds, {negate});
  std::cout << transformds.size() << "\n"; // 10
  std::cout << allClose(transformds.get(5)[0], -ds->get(5)[0]) << "\n"; // 1
  \endcode
 */
class FL_API TransformDataset : public Dataset {
 public:
  /**
   * Creates a `TransformDataset`.
   * @param[in] dataset The underlying dataset.
   * @param[in] transformfns The mappings used to transform the values.
   * If a `TransformFunction` is null then the corresponding value is not
   * transformed.
   */
  TransformDataset(
      std::shared_ptr<const Dataset> dataset,
      const std::vector<TransformFunction>& transformfns);

  int64_t size() const override;

  std::vector<Tensor> get(const int64_t idx) const override;

 private:
  std::shared_ptr<const Dataset> dataset_;
  const std::vector<TransformFunction> transformFns_;
};
} // namespace fl
