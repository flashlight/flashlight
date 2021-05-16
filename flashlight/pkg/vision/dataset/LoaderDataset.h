/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/dataset/datasets.h"

namespace fl {
namespace ext {
namespace image {

/*
 * Small generic utility class for loading data from a vector of type T into an
 * vector of arrayfire arrays
 */
template <typename T>
class LoaderDataset : public fl::Dataset {
 public:
  using LoadFunc = std::function<std::vector<af::array>(const T&)>;

  LoaderDataset(const std::vector<T>& list, LoadFunc loadfn)
      : list_(list), loadfn_(loadfn) {}

  std::vector<af::array> get(const int64_t idx) const override {
    return loadfn_(list_[idx]);
  }

  int64_t size() const override {
    return list_.size();
  }

 private:
  std::vector<T> list_;
  LoadFunc loadfn_;
};

} // namespace image
} // namespace ext
} // namespace fl
