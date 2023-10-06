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
 * Dataset created by unpacking tensors along the last non-singleton dimension.
 *
 * The size of the dataset is determined by the size along that dimension.
 * Hence, it must be the same across all `int64_t`s in the input.
 *
 * Example:
  \code{.cpp}
  Tensor tensor1 = fl::rand({5, 4, 10});
  Tensor tensor2 = fl::rand({7, 10});
  TensorDataset ds({tensor1, tensor2});

  std::cout << ds.size() << "\n"; // 10
  std::cout << ds.get(0)[0].shape() << "\n"; // 5 4
  std::cout << ds.get(0)[1].shape() << "\n"; // 7 1
  \endcode
 */
class FL_API TensorDataset : public Dataset {
 public:
  /**
   * Creates a `TensorDataset` by unpacking the input tensors.
   * @param[in] datatensors A vector of tensors, which will be
   * unpacked along their last non-singleton dimensions.
   */
  explicit TensorDataset(const std::vector<Tensor>& datatensors);

  int64_t size() const override;

  std::vector<Tensor> get(const int64_t idx) const override;

 private:
  std::vector<Tensor> dataTensors_;
  int64_t size_{0};
};
} // namespace fl
