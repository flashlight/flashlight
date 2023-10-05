/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/dataset/datasets.h"

namespace fl {
namespace pkg {
namespace vision {

using TransformAllFunction =
    std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;

/*
 * A view into a dataset where all arrays are transformed using the same
 * function.
 *
 * This is valuable for object detection, because transforms are not
 * independent of each other. For example, a random crop must crop the iamge
 * but then also adjust the bounding boxes accordinly
 */
class TransformAllDataset : public Dataset {
 public:
  TransformAllDataset(
      std::shared_ptr<const Dataset> dataset,
      TransformAllFunction fn)
      : dataset_(dataset), fn_(fn) {}

  std::vector<Tensor> get(const int64_t idx) const override {
    return fn_(dataset_->get(idx));
  }

  int64_t size() const override {
    return dataset_->size();
  }

 private:
  std::shared_ptr<const Dataset> dataset_;
  const TransformAllFunction fn_;
};

} // namespace vision
} // namespace pkg
} // namespace fl
