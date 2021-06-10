/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Index.h"

namespace fl {

range::range(int idx) : range(0, idx) {}

range::range(int start, int end) : range(start, end, /* stride */ 1) {}

range::range(int start, int end, int stride)
    : start_(start), end_(end - 1), stride_(stride) {}

int range::start() const {
  return start_;
}

int range::end() const {
  return end_;
}

int range::stride() const {
  return stride_;
}

bool range::operator==(const range& other) const {
  return start_ == other.start() && end_ == other.end() &&
      stride_ == other.stride();
}

bool range::operator!=(const range& other) const {
  return !this->operator==(other);
}

Index::Index(const Tensor& tensor)
    : type_(detail::IndexType::Tensor), index_(tensor) {}

Index::Index(const range& range)
    : type_(detail::IndexType::Range), index_(range) {}

Index::Index(const int idx) : type_(detail::IndexType::Literal), index_(idx) {}

detail::IndexType Index::type() const {
  return type_;
}

bool Index::isSpan() const {
  if (type_ != detail::IndexType::Range) {
    return false;
  }
  return get<range>() == fl::span;
}

} // namespace fl
