/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iterator>

namespace fl {
namespace detail {

/**
 * STL style iterator class to easily iterate over a dataset.
 *
 * Example:
  \ code{.cpp}
  Tensor tensor = fl::rand({1, 2, 3});
  TensorDataset tensords(std::vector<Tensor>{tensor});
  for (auto& sample : tensords) {
    // do something
  }
  \endcode
 */
template <typename D, typename F>
class DatasetIterator {
 protected:
  D* dataset_;
  int64_t idx_;
  F buffer_;

 public:
  // DatasetIterator traits, previously from std::iterator.
  using value_type = F;
  using reference = F&;
  using pointer = F*;
  using iterator_category = std::forward_iterator_tag;

  // Default constructible.
  DatasetIterator() : dataset_(nullptr), idx_(-1) {}

  explicit DatasetIterator(D* dataset)
      : dataset_(dataset), idx_(dataset_->size() > 0 ? 0 : -1) {}

  // Dereferencable
  reference operator*() {
    buffer_ = dataset_->get(idx_);
    return buffer_;
  }

  // Pre- and post-incrementable.
  DatasetIterator& operator++() {
    if (++idx_ >= dataset_->size()) {
      idx_ = -1;
    }
    return *this;
  }

  DatasetIterator operator++(int) {
    DatasetIterator tmp(*this);
    if (++idx_ >= dataset_->size())
      idx_ = -1;
    return tmp;
  }

  // Equality / inequality.
  bool operator==(const DatasetIterator& that) const {
    return (idx_ == that.idx_);
  }

  bool operator!=(const DatasetIterator& that) const {
    return (idx_ != that.idx_);
  }
};

} // namespace detail
} // namespace fl
