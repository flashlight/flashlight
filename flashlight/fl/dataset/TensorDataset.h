/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
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
  af::array tensor1 = af::randu(5, 4, 10);
  af::array tensor2 = af::randu(7, 10);
  TensorDataset ds({tensor1, tensor2});

  std::cout << ds.size() << "\n"; // 10
  std::cout << ds.get(0)[0].dims() << "\n"; // 5 4 1 1
  std::cout << ds.get(0)[1].dims() << "\n"; // 7 1 1 1
  \endcode
 */
class TensorDataset : public Dataset {
 public:
  /**
   * Creates a `TensorDataset` by unpacking the input tensors.
   * @param[in] datatensors A vector of tensors, which will be
   * unpacked along their last non-singleton dimensions.
   */
  explicit TensorDataset(const std::vector<af::array>& datatensors);

  int64_t size() const override;

  std::vector<af::array> get(const int64_t idx) const override;

 private:
  std::vector<af::array> dataTensors_;
  int64_t size_;
};
} // namespace fl
