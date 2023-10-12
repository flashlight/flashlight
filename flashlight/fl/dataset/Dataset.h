/*
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/dataset/DatasetIterator.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/**
 * Abstract class representing a dataset: a mapping index -> sample,
 * where a sample is a vector of `Tensor`s
 *
 * Can be extended to concat, split, batch, resample, etc. datasets.
 *
 * A `Dataset` can either own its data directly, or through `shared_ptr`
 * ownership of underlying `Dataset`s.
 */
class FL_API Dataset {
 public:
  /**
   * A bijective mapping of dataset indices \f$[0, n) \to [0, n)\f$.
   */
  using PermutationFunction = std::function<int64_t(int64_t)>;

  /**
   * A function to transform an array.
   */
  using TransformFunction = std::function<Tensor(const Tensor&)>;

  /**
   * A function to load data from a file into an array.
   */
  using LoadFunction = std::function<Tensor(const std::string&)>;

  /**
   * A function to pack arrays into a batched array.
   */
  using BatchFunction = std::function<Tensor(const std::vector<Tensor>&)>;

  /**
   * A function to transform data from host to array.
   */
  using DataTransformFunction =
      std::function<Tensor(void*, fl::Shape, fl::dtype)>;

  /**
   * @return The size of the dataset.
   */
  virtual int64_t size() const = 0;

  /**
   * @param[in] idx Index of the sample in the dataset. Must be in [0, size()).
   * @return The sample fields (a `std::vector<Tensor>`).
   */
  virtual std::vector<Tensor> get(const int64_t idx) const = 0;

  virtual ~Dataset() = default;

  // Setup iterators
  using iterator = detail::DatasetIterator<Dataset, std::vector<Tensor>>;

  iterator begin() {
    return iterator(this);
  }

  iterator end() {
    return iterator();
  }

 protected:
  void checkIndexBounds(int64_t idx) const {
    if (!(idx >= 0 && idx < size())) {
      throw std::out_of_range("Dataset idx out of range");
    }
  }
};

} // namespace fl
