/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iterator>

#include <arrayfire.h>

namespace fl {
namespace detail {
// class DatasetIterator
//
// STL style iterator class to easily iterate over a dataset
// Example usage:
//    af::array tensor = af::randu(1, 2, 3);
//    TensorDataset tensords(std::vector<af::array>{tensor});
//    for (auto& sample : tensords) {
//      // do something
//    }

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
